"""球检测 + 球台区域门控 + 触台判定（参赛者代码区）。

定位：官方 ONNX 权重（weights/ball.onnx）+ TensorRT，保持官方解码不变。
球台门控：预训练 ResNet18 ONNX（weights/table_region.onnx）逐帧估计球台框；
触点超出球台预测框时过滤。空掩码或异常面积时保留原触点。
事件：ContactTracker 在 ±window 帧上打分 y 方向局部极大值，幅度用对称窗口
测量以抵抗检测噪声；判定延迟到后续顶点足够远时再输出，两个近邻顶点竞争时
保留更强者（等价离线 NMS）。触点自身置信度要求 0.80，两侧参考帧要求
0.70；置信度在 0.75 至 0.80 之间的触点，仅在相邻两帧均达到 0.80 且
共同确认反弹时补回，之后仍须通过窗口幅度、跳变、NMS 和球台门控。
透视下未形成 y 极大值时，用五帧高置信度轨迹的向上速度突变补充检测；
外围参考帧缺失时允许相邻帧短窗口补救，并检查单帧跳变以排除串到其他物体。
单个外围参考点较弱时可补充低优先级候选；选定候选后排除停滞和明确水平反向。
视频结束时保留原输出，再补充未被覆盖的可靠候选；短缺帧仅用于补全参考轨迹，
此补救分支输出实际检测到的球心，并通过球台门控及时间/空间去重。
最终对高速轨迹拟合连续触台时刻并细化坐标；补充可靠的高速触点，
并排除以水平方向速度突变为主的反向事件。全部落点在视频结束时统一输出。

与官方参考实现的差异：幅度窗口 1 → 2 帧；落点最小间隔 10 → 7 帧；
下落/反弹门控 2.0/1.0 → 0.5/0.5；新增轨迹跳变上限与延迟决策 NMS。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.types import Landing, RunContext, VideoMeta

from .base_solution import BaseSolution
from .ball_detector import BallDetector
from .contact_tracker import ContactTracker
from .contact_recovery import ContactRecovery
from .contact_refinement import ContactRefinement
from .table_region_detector import TableRegionDetector

RULE = dict(window=2, conf_th=0.70, vertex_conf=0.80, rescue_vertex_conf=0.75, min_gap=7,
            min_fall=0.5, min_rise=0.5, max_jump=110.0,
            velocity_impulse=10.0, max_step=220.0, short_window_fallback=True,
            secondary_impulse=True, min_contact_motion=1.0, reject_contact_reversal=True)
TABLE_MARGIN = 0.0


class Solution(BaseSolution):
    """触台判定：球心 y 的局部极大 + 对称窗口幅度门控 + 延迟 NMS。"""

    def prepare(self, context: RunContext) -> None:
        # 1. 权重：只带 ONNX，TRT engine 现场构建（首跑约 5 分钟，之后从缓存秒级加载）
        onnx_path = Path(__file__).resolve().parent / "weights" / "ball.onnx"
        assert onnx_path.exists(), f"缺少权重: {onnx_path}"

        # 2. TRT 转换：core.trt.load_or_build 自动处理缓存（按版本/GPU/ONNX 哈希分目录）
        from core.trt import load_or_build
        trt_cache = Path(context.output_dir) / "trt_cache"
        model, how = load_or_build(
            onnx_path, trt_cache, input_name="input",
            shape=(1, 9, 288, 512), log=print)

        # 3. 检测器：只管预处理 + 推理 + 解码，推理函数由 prepare 注入
        self.det = BallDetector(model.infer, onnx_path)

        # 另一个静态 ResNet18 分支给出当前帧球台框；输出异常时不做门控。
        table_onnx = Path(__file__).resolve().parent / "weights" / "table_region.onnx"
        assert table_onnx.exists(), f"缺少权重: {table_onnx}"
        table_model, table_how = load_or_build(
            table_onnx, trt_cache, input_name="image", shape=(1, 3, 192, 320), log=print)
        self.table = TableRegionDetector(table_model.infer)

        # 4. warmup：TRT 首次推理会触发 kernel JIT 编译，跑一次 dummy 把 kernel
        #    都预热好，避免第一视频第一帧把这个时间算进 infer_sec
        self.det.detect(np.zeros((288, 512, 3), np.uint8), 512, 288)

        # 5. 回填 backend 给 run_status（让平台看到是 built 还是 loaded）
        context.info["backend"] = f"tensorrt_ball_{how}_table_{table_how}"

        self.tracker = ContactTracker(**RULE)
        self.recovery = ContactRecovery(RULE, table_margin=TABLE_MARGIN)
        self.refinement = ContactRefinement(RULE, table_margin=TABLE_MARGIN)
        self.output_hits = []
        self.table_boxes = {}

    def reset(self, video_meta: VideoMeta) -> None:
        self.det.reset()
        self.tracker.reset()
        self.recovery.reset()
        self.output_hits.clear()
        self.table_boxes.clear()

    def process_frame(self, frame_rgb, frame_meta) -> list[Landing]:
        fid = frame_meta["frame_id"]
        table_box = self.table.box(frame_rgb, frame_meta["width"], frame_meta["height"])
        if table_box is not None:
            self.table_boxes[fid] = table_box
        det = self.det.detect(frame_rgb, frame_meta["width"], frame_meta["height"])
        if det is not None:
            # ch1 解码的是三帧栈的中间帧（滞后 1 帧）：当前帧拿到的是 fid-1 的球位
            self.tracker.add(fid - 1, (det["x"], det["y"], det["conf"]))
            self.recovery.record(fid - 1, (det["x"], det["y"], det["conf"]))
        # 已知的最大球位帧是 fid-1；判定顶点需要它之后 window 帧，故推进到 fid-1
        self.output_hits.extend(self._gate(self.tracker.advance(fid - 1)))
        return []

    def _gate(self, hits) -> list[dict]:
        output = []
        for hit in hits:
            box = self.table_boxes.get(hit["frame_id"])
            # A missing or implausible mask is uncertain; preserve the candidate.
            if box is None or (box[0] - TABLE_MARGIN <= hit["x"] <= box[2] + TABLE_MARGIN
                               and box[1] - TABLE_MARGIN <= hit["y"] <= box[3] + TABLE_MARGIN):
                output.append(dict(hit))
                self.recovery.remember(hit)
        return output

    def finish_video(self, video_meta: VideoMeta) -> list[Landing]:
        # 尾帧 flush：把缓冲中的候选输出，避免丢视频末尾落点
        self.output_hits.extend(self._gate(self.tracker.flush()))
        self.output_hits.extend(self._gate(self.recovery.recover(self.table_boxes)))
        output = self.refinement.refine(self.output_hits, self.recovery.points, self.table_boxes)
        return [Landing(hit['frame_id'], hit['x'], hit['y']) for hit in output]

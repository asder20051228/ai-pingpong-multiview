# 4070 TensorRT 正式入口复测（2026-10-18）

## 本次结果

两组均完成全部 15 个视频、2109 帧，框架完整性和输出格式检查全部通过。以下均为精度分，未计速度分。

| 方法 | 宽松口径 /75 | 严格口径 /75 |
| --- | ---: | ---: |
| 关闭球台门控 | 46.43 | 45.54 |
| 修复后开启门控 | **50.50** | **49.46** |
| 门控增益 | +4.07 | +3.92 |

开启门控的 76 条预测全部包含在关闭组的 88 条预测中；门控恰好去除 12 条记录。两种评分口径下 TP 均未减少，减少的均为 FP。

相比此前修复后的 ORT+OpenCV 路径，TensorRT+ffmpeg CUDA 路径的宽松分数由 52.1361 降至 50.5027：0° TP 不变、FP 多 1；45° TP 从 22 降至 20、FP 从 11 增至 12；90° 结果不变。关闭门控时也存在这些差异，因此不能归因于球台门控误删。FP16 推理和 GPU 解码均随运行路径改变，本次未进一步隔离这两项因素，不能断言具体根因。

## 复现

服务器上的独立环境：`/home/aiinfer/pingpang_training/trt_env_setup/env`。

执行：

```bash
bash /home/aiinfer/pingpang_training/official_eval/trt_normfix_20261018/run_formal.sh
```

脚本先校验官方框架完整性和依赖版本，再直接执行未经修改的 `pingpang/run.py --decoder gpu`。球检测和球台分割均使用 `core.trt` 构建的 TensorRT 引擎。首次执行会构建引擎；后续复用缓存。

门控开启使用正式参赛类 `participant.solution:Solution`。关闭对照使用一个子类，仅覆盖 `_gate`，直接返回候选触点，其余模型、规则、输入和引擎保持一致。该对照用于精度比较，不用于衡量关闭球台模型能带来的速度收益。

## 结果文件

- `report.json`：分数、逐视角 TP/FP/FN、运行状态、依赖版本和文件 SHA256。
- `gate/predictions.jsonl`、`no_gate/predictions.jsonl`：两组预测。
- `environment.log`：比赛依赖契约检查。
- `integrity.log`、`integrity_after.log`：评测前后的框架完整性检查。
- `gate_console.log`、`no_gate_console.log`：TensorRT 引擎构建及推理日志。
- `score_runs.py`：沿用上一轮的公开集评分代码；仅更换待评分输出路径。

宽松口径忽略匹配到 `valid=false` 标注的预测，严格口径将未匹配有效 GT 的预测计为 FP。两者都采用帧号 ±1、欧氏距离 50px、一对一匹配和三视角 F1×25 求和。旧报告先把各视角 F1 四舍五入到四位再求和，可能产生 0.01 分的显示差异。

此复测使用 RTX 4070 上的公开验证集，未计速度分。宿主系统为 Ubuntu 24.04，独立环境匹配比赛锁定的 Python 3.10 / PyTorch 2.1.2+cu118 / TensorRT 8.6.1 / NumPy 1.26.4 / ORT 1.18.1；并非组委会 Ubuntu 22.04 Docker 镜像，也不是官方隐藏测试集榜单成绩。

"""Bounded decode/inference overlap, optional table inference on visible GPU 1.

The official runner/decoder stay unchanged. All queued work is joined before
finish_video returns, so official wall time includes every frame's work.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import numpy as np
import torch
from .adaptive_solution import Solution as Frozen86


def choose_devices(visible_count):
    if visible_count < 1:
        raise RuntimeError('TensorRT solution requires at least one visible CUDA GPU')
    return (0, 1) if visible_count >= 2 else (0, 0)


class BranchSolution(Frozen86):
    force_branch_test = False  # Diagnostic subclass only; never invent a GPU.

    def prepare(self, context):
        count = torch.cuda.device_count()
        self.ball_device, self.table_device = choose_devices(count)
        with torch.cuda.device(self.ball_device):
            super().prepare(context)
        self.branch_pool = None
        if self.table_device != self.ball_device or self.force_branch_test:
            self.branch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='table')
            self.branch_pool.submit(self._prepare_table, context).result()
        context.info['gpu_execution'] = dict(
            visible_count=count, used_count=min(count, 2),
            ball_device=self.ball_device, table_device=self.table_device,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            branch_parallel=self.branch_pool is not None,
            same_gpu_branch_diagnostic=self.force_branch_test,
            devices=[torch.cuda.get_device_name(i) for i in range(min(count, 2))])
        print('GPU_EXECUTION', context.info['gpu_execution'], flush=True)

    def _prepare_table(self, context):
        from core.trt import load_or_build
        from .frozen85.participant import solution as base
        from .frozen85.participant.table_region_detector import TableRegionDetector
        onnx = Path(base.__file__).parent / 'weights' / 'table_region.onnx'
        # Same GPU model can reuse its exact frozen engine. Other models get an
        # isolated cache, never deserialize an engine built for the first GPU.
        cache = Path(context.output_dir) / 'trt_cache'
        if torch.cuda.get_device_name(self.table_device) != torch.cuda.get_device_name(0):
            cache = cache / ('device_' + str(self.table_device))
        with torch.cuda.device(self.table_device):
            self.table_model, _ = load_or_build(onnx, cache, 'image', (1, 3, 192, 320))
            self.table = TableRegionDetector(self.table_model.infer)
            self.table.box(np.zeros((192, 320, 3), np.uint8), 320, 192)

    def _table_observation(self, rgb, meta):
        with torch.cuda.device(self.table_device):
            return self.table.box(rgb, meta['width'], meta['height'])

    def process_frame(self, rgb, meta):
        with torch.cuda.device(self.ball_device):
            if self.branch_pool is None:
                return super().process_frame(rgb, meta)
            self.active_frame = meta['frame_id']
            future = self.branch_pool.submit(self._table_observation, rgb, meta)
            detect = self.det.detect
            box = self.table.box
            try:
                ball_value = detect(rgb, meta['width'], meta['height'])
                table_value = future.result()
            except BaseException:
                future.result()  # Do not leave table work racing the next reset.
                raise
            # Reuse the complete frozen per-frame state transition, with the two
            # already-computed observations. No second buffer advance/inference.
            self.det.detect = lambda *args: ball_value
            self.table.box = lambda *args: table_value
            try:
                return super().process_frame(rgb, meta)
            finally:
                self.det.detect = detect
                self.table.box = box

    def close(self):
        if getattr(self, 'branch_pool', None) is not None:
            self.branch_pool.shutdown(wait=True, cancel_futures=True)


class Solution:
    core_class = BranchSolution

    def prepare(self, context):
        self.pending = deque()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ordered-inference')
        self.core = self.core_class()
        try:
            self.worker.submit(self.core.prepare, context).result()
        except BaseException:
            self.close()
            raise
        context.info['pipeline'] = dict(max_pending_frames=2, ordered=True, finish_joins=True)

    def _drain(self):
        outputs = []
        while self.pending:
            outputs.extend(self.pending.popleft().result() or [])
        return outputs

    def reset(self, meta):
        self._drain()
        self.worker.submit(self.core.reset, meta).result()

    def process_frame(self, rgb, meta):
        outputs = []
        if len(self.pending) >= 2:
            outputs.extend(self.pending.popleft().result() or [])
        self.pending.append(self.worker.submit(self.core.process_frame, rgb.copy(), dict(meta)))
        return outputs

    def finish_video(self, meta):
        output = self._drain()
        output.extend(self.worker.submit(self.core.finish_video, meta).result() or [])
        return output

    def close(self):
        if hasattr(self, 'worker'):
            self.worker.shutdown(wait=True, cancel_futures=True)
        if hasattr(self, 'core'):
            self.core.close()


class SameGpuBranch(BranchSolution):
    force_branch_test = True


class SameGpuBranchDiagnostic(Solution):
    core_class = SameGpuBranch

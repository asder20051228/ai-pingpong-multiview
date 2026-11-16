"""Recheck only confidence-boundary ball heatmaps with strict FP32."""
import json
import numpy as np
from pathlib import Path
from .final_solution import Solution as FrozenColor
from .rescue_solution import RescueMixin
from .precision_solution import strict_model
class AdaptiveMixin:
    def prepare(self,context):
        super().prepare(context)
        self.precise_model=strict_model(context)
        self.precise_model.infer(np.zeros((1,9,288,512),np.float32))
        self.precision_dir=Path(context.output_dir)/'precision_audit';self.precision_dir.mkdir(exist_ok=True)
        original=self.det._infer
        def infer(x):
            y=original(x);confidence=float(y[0,1].max())
            if abs(confidence-.80)<=.005:
                y=self.precise_model.infer(x)
                self.precision_rows.append(dict(input_frame=self.active_frame,fp16_conf=confidence,fp32_conf=float(y[0,1].max())))
            self.latest_heat=y[0]
            return y
        self.det._infer=infer
        context.info['backend']='fp16_ball_strict_fp32_recheck_0.795_0.805_table_fp16'
    def reset(self,meta):
        super().reset(meta);self.precision_rows=[];self.active_frame=None
    def process_frame(self,rgb,meta):
        self.active_frame=meta['frame_id'];return super().process_frame(rgb,meta)
    def finish_video(self,meta):
        output=super().finish_video(meta)
        (self.precision_dir/(meta.video_id+'.json')).write_text(json.dumps(self.precision_rows,indent=2))
        return output
class ControlSolution(AdaptiveMixin,FrozenColor):pass
class Solution(AdaptiveMixin,RescueMixin,FrozenColor):pass

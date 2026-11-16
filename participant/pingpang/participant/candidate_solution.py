from pathlib import Path
import json
from core.types import Landing
from .frozen85.future_solution import Solution as FrozenSolution
from .event_rules import transform
class Solution(FrozenSolution):
    mode='combined'
    def prepare(self,context):
        super().prepare(context);self.changes_dir=Path(context.output_dir)/'event_changes';self.changes_dir.mkdir(exist_ok=True)
    def finish_video(self,meta):
        baseline=super().finish_video(meta)
        original=dict(self.recovery.points)
        from .frozen85.experiment_solution import fuse
        points,accepted,_=fuse(original,self.candidates,meta.width,meta.height,True)
        output,changes=transform([h.to_dict() for h in baseline],points,accepted,self.table_boxes,self.mode)
        (self.changes_dir/(meta.video_id+'.json')).write_text(json.dumps(changes,indent=2))
        return [Landing(h['frame_id'],h['x'],h['y']) for h in output]
class ShortSolution(Solution):mode='short'
class HorizontalSolution(Solution):mode='horizontal'
class MicroSolution(Solution):mode='micro'

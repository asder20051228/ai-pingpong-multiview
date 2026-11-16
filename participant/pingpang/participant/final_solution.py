"""Independent experiment: trajectory rules and scene-relative table support."""
from pathlib import Path
import json
from .candidate_solution import Solution as EventSolution
from .color_probe_solution import color_map, support_at

class Solution(EventSolution):
    def prepare(self, context):
        super().prepare(context)
        self.color_dir = Path(context.output_dir) / 'color_evidence'
        self.color_dir.mkdir(exist_ok=True)

    def reset(self, meta):
        super().reset(meta)
        self.color_maps = {}

    def process_frame(self, rgb, meta):
        result = super().process_frame(rgb, meta)
        frame = meta['frame_id']
        self.color_maps[frame] = color_map(rgb, self.table_boxes.get(frame))
        return result

    def finish_video(self, meta):
        candidates = super().finish_video(meta)
        rows = []
        output = []
        for hit in candidates:
            evidence = support_at(self.color_maps, hit.to_dict(), meta.width, meta.height)
            rows.append(dict(hit=hit.to_dict(), **evidence))
            if not evidence['veto']:
                output.append(hit)
        (self.color_dir / (meta.video_id + '.json')).write_text(json.dumps(rows, indent=2))
        return output

class ColorOnlySolution(Solution):
    mode = 'baseline'

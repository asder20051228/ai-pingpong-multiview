"""Independent numerical-boundary rescue. No video IDs, frame IDs or GT access."""
import json,math
from pathlib import Path
from core.types import Landing
from .final_solution import Solution as FrozenColor
from .precision_solution import Solution as StrictColor
from .frozen85.participant.solution import Solution as ReplaySolution,RULE
from .frozen85.participant.contact_tracker import ContactTracker
from .frozen85.participant.contact_recovery import ContactRecovery
from .frozen85.participant.contact_refinement import ContactRefinement
from .frozen85.experiment_solution import fuse
from .event_rules import transform,conflicts,inside
from .color_probe_solution import support_at

def proposal(points,accepted,boxes,n_frames,confidence_band,shallow_band):
    rule=dict(RULE)
    if confidence_band:rule['vertex_conf']=.795
    if shallow_band:rule['min_fall']=0.;rule['min_rise']=0.
    s=ReplaySolution();s.tracker=ContactTracker(**rule);s.recovery=ContactRecovery(rule)
    s.refinement=ContactRefinement(rule);s.output_hits=[];s.table_boxes=boxes
    for fid in range(n_frames):
        f=fid-1;p=points.get(f)
        if p is not None:s.tracker.add(f,p);s.recovery.record(f,p)
        s.output_hits.extend(s._gate(s.tracker.advance(f)))
    hits=[h.to_dict() for h in s.finish_video(None)]
    hits,_=transform(hits,points,accepted,boxes,'combined')
    output=[]
    for h in hits:
        f=h['frame_id'];ps=[points.get(i) for i in (f-1,f,f+1)]
        if any(p is None for p in ps):continue
        a,b,c=ps
        # Tolerance applies to strong bounces near a recovered observation only.
        if b[2]<.8 or min(a[2],c[2])<.795:continue
        if not any(i in accepted for i in (f-1,f,f+1)):continue
        fall=b[1]-a[1];rise=b[1]-c[1];impulse=fall+rise
        if min(fall,rise)<0 or max(fall,rise)<10 or impulse<10:continue
        dx0=b[0]-a[0];dx1=c[0]-b[0]
        if abs(dx1-dx0)>impulse or (dx0*dx1<0 and min(abs(dx0),abs(dx1))>2):continue
        if min(math.dist(a[:2],b[:2]),math.dist(b[:2],c[:2]))<1:continue
        if max(math.dist(a[:2],b[:2]),math.dist(b[:2],c[:2]))>220:continue
        if inside(h,boxes):output.append(h)
    return output

class RescueMixin:
    confidence_band=True
    shallow_band=True
    def prepare(self,context):
        super().prepare(context)
        self.rescue_dir=Path(context.output_dir)/'boundary_rescue';self.rescue_dir.mkdir(exist_ok=True)
    def finish_video(self,meta):
        baseline=super().finish_video(meta)
        points,accepted,_=fuse(dict(self.recovery.points),self.candidates,meta.width,meta.height,True)
        proposals=proposal(points,accepted,self.table_boxes,meta.n_frames,self.confidence_band,self.shallow_band)
        output=[h.to_dict() for h in baseline];audit=[]
        for h in proposals:
            if any(conflicts(h,old) for old in output):continue
            support=support_at(self.color_maps,h,meta.width,meta.height)
            audit.append(dict(hit=h,color=support,accepted=not support['veto']))
            if not support['veto']:output.append(h)
        (self.rescue_dir/(meta.video_id+'.json')).write_text(json.dumps(audit,indent=2))
        return [Landing(h['frame_id'],h['x'],h['y']) for h in output]
class ConfidenceSolution(RescueMixin,FrozenColor):shallow_band=False
class ShallowSolution(RescueMixin,FrozenColor):confidence_band=False
class Solution(RescueMixin,FrozenColor):pass
class StrictSolution(RescueMixin,StrictColor):pass

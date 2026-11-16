"""Isolated temporal-channel rescue; no labels or video-specific rules."""
from pathlib import Path
import json, math
import cv2
import numpy as np
from .participant.solution import Solution as FrozenSolution, RULE
from .participant.contact_tracker import ContactTracker
from .participant.contact_recovery import ContactRecovery
from .participant.contact_refinement import ContactRefinement

def components(heat,width,height):
    mask=(heat>.7).astype(np.uint8)
    if not mask.any():return []
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    result=[]
    for label in range(1,count):
        x,y,w,h,area=stats[label]
        region=heat[y:y+h,x:x+w]; keep=labels[y:y+h,x:x+w]==label
        yy,xx=np.nonzero(keep);weights=region[yy,xx].astype(np.float64)
        mass=float(weights.sum())
        result.append(dict(x=float(((xx+x)*weights).sum()/mass*width/heat.shape[1]),
                           y=float(((yy+y)*weights).sum()/mass*height/heat.shape[0]),
                           conf=float(weights.max()),mass=mass,pixels=int(area)))
    return sorted(result,key=lambda p:(p['conf'],p['mass']),reverse=True)[:4]

def continuity(frame,p,original,width,height):
    sx=width/1920.; sy=height/1080.
    left=[f for f in range(frame-1,max(-1,frame-5),-1) if f in original and original[f][2]>=.8]
    right=[f for f in range(frame+1,frame+5) if f in original and original[f][2]>=.8]
    if left and right:
        a,b=left[0],right[0]
        if b-a>6:return False,'reference_gap'
        pa,pb=original[a],original[b];fraction=(frame-a)/(b-a)
        ex=pa[0]+fraction*(pb[0]-pa[0]);ey=pa[1]+fraction*(pb[1]-pa[1])
        vx=(pb[0]-pa[0])/(b-a)
        if abs(p['x']-ex)>24*sx+.2*abs(vx):return False,'horizontal_residual'
        if abs(p['y']-ey)>max(32*sy,.75*abs(pb[1]-pa[1])):return False,'vertical_residual'
        for f,pt in [(a,pa),(b,pb)]:
            if math.hypot((p['x']-pt[0])/sx,(p['y']-pt[1])/sy)/abs(frame-f)>RULE['max_step']:
                return False,'step_limit'
        return True,'two_sided'
    same=left or right
    if len(same)<2:return False,'insufficient_original_support'
    a,b=same[:2]
    if abs(a-frame)>2 or p['conf']<.85:return False,'one_sided_confidence_or_distance'
    pa,pb=original[a],original[b];vx=(pb[0]-pa[0])/(b-a);vy=(pb[1]-pa[1])/(b-a)
    ex=pa[0]+vx*(frame-a);ey=pa[1]+vy*(frame-a)
    if abs(p['x']-ex)>24*sx+.2*abs(vx):return False,'horizontal_residual'
    if abs(p['y']-ey)>max(32*sy,1.5*abs(vy*(frame-a))):return False,'vertical_residual'
    if math.hypot((p['x']-pa[0])/sx,(p['y']-pa[1])/sy)/abs(frame-a)>RULE['max_step']:
        return False,'step_limit'
    return True,'one_sided'

def fuse(original,candidates,width,height,constrained):
    fused=dict(original);accepted={};rejected={}
    for f,choices in sorted(candidates.items()):
        if f in original:continue
        decisions=[]
        for p in sorted(choices,key=lambda p:(p['conf'],p['mass']),reverse=True):
            ok,reason=continuity(f,p,original,width,height) if constrained else (True,'unconstrained')
            decisions.append(dict(candidate=p,accepted=ok,reason=reason))
            if ok:
                fused[f]=(p['x'],p['y'],p['conf']);accepted[f]=dict(p,reason=reason);break
        rejected[f]=decisions
    return fused,accepted,rejected

def replay(points,boxes,n_frames):
    # Same stream ordering, gating, recovery and refinement as frozen Solution.
    s=FrozenSolution();s.tracker=ContactTracker(**RULE);s.recovery=ContactRecovery(RULE)
    s.refinement=ContactRefinement(RULE);s.output_hits=[];s.table_boxes=boxes
    for fid in range(n_frames):
        frame=fid-1;p=points.get(frame)
        if p is not None:s.tracker.add(frame,p);s.recovery.record(frame,p)
        s.output_hits.extend(s._gate(s.tracker.advance(frame)))
    return s.finish_video(None)

class Solution(FrozenSolution):
    constrained=True
    def prepare(self,context):
        super().prepare(context)
        self.audit_dir=Path(context.output_dir)/'fusion_audit';self.audit_dir.mkdir(exist_ok=True)
        original=self.det._infer
        def infer(x):
            output=original(x);self.latest_heat=output[0];return output
        self.det._infer=infer

    def reset(self,meta):
        super().reset(meta)
        self.previous_newest=None;self.latest_heat=None;self.candidates={};self.n_frames=meta.n_frames

    def process_frame(self,frame_rgb,frame_meta):
        super().process_frame(frame_rgb,frame_meta)
        fid=frame_meta['frame_id'];w=frame_meta['width'];h=frame_meta['height']
        original=self.recovery.points
        # ch2 from preceding window refers to fid-1, now known to be missing in ch1.
        if fid-1>=0 and fid-1 not in original and self.previous_newest is not None:
            self.candidates.setdefault(fid-1,[]).extend(dict(p,channel=2) for p in components(self.previous_newest,w,h))
        # ch0 from this window refers to fid-2, supplying one further frame of context.
        if fid-2>=0 and fid-2 not in original:
            self.candidates.setdefault(fid-2,[]).extend(dict(p,channel=0) for p in components(self.latest_heat[0],w,h))
        self.previous_newest=np.array(self.latest_heat[2],copy=True)
        return []

    def finish_video(self,meta):
        original=dict(self.recovery.points)
        fused,accepted,rejected=fuse(original,self.candidates,meta.width,meta.height,self.constrained)
        output=replay(fused,self.table_boxes,meta.n_frames)
        record=dict(video_id=meta.video_id,constrained=self.constrained,original_points=original,
                    accepted=accepted,decisions=rejected,table_boxes=self.table_boxes,
                    predictions=[h.to_dict() for h in output])
        (self.audit_dir/(meta.video_id+'.json')).write_text(json.dumps(record),encoding='utf-8')
        print('FUSION',meta.video_id,'observed',len(original),'added',len(accepted),'constrained',self.constrained,flush=True)
        return output

class UnconstrainedSolution(Solution):
    constrained=False

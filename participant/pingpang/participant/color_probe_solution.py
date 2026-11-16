"""Scene-relative color evidence. Uncertain templates never veto a contact."""
from pathlib import Path
import json,cv2,numpy as np
from .frozen85.future_solution import Solution as FrozenSolution
from .frozen85.experiment_solution import fuse
from .event_rules import transform
def color_map(rgb,box):
    if box is None:return None
    h,w=rgb.shape[:2];im=cv2.resize(rgb,(320,180),interpolation=cv2.INTER_LINEAR)
    hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV);x0,y0,x1,y1=box
    # Central/lower rectangle is used only to infer a dominant scene color.
    a,b=int((x0+.25*(x1-x0))*320/w),int((x0+.75*(x1-x0))*320/w)
    c,d=int((y0+.35*(y1-y0))*180/h),int((y0+.8*(y1-y0))*180/h)
    patch=hsv[max(0,c):min(180,d),max(0,a):min(320,b)]
    if not patch.size:return None
    valid=(patch[:,:,1]>=25)&(patch[:,:,2]>=40)
    if valid.sum()<100:return None
    hues=patch[:,:,0][valid];hist=np.bincount(hues//5,minlength=36)
    smooth=hist+np.roll(hist,1)+np.roll(hist,-1)
    peak=int(smooth.argmax())*5+2
    delta=np.minimum(abs(hues.astype(float)-peak),180-abs(hues.astype(float)-peak))
    if np.mean(delta<=12)<.55:return None
    sat=float(np.median(patch[:,:,1][valid][delta<=12]))
    delta=np.minimum(abs(hsv[:,:,0].astype(float)-peak),180-abs(hsv[:,:,0].astype(float)-peak))
    support=(delta<=12)&(hsv[:,:,1]>=max(10,sat*.25))&(hsv[:,:,2]>=30)
    return dict(mask=support,hue=peak,saturation=sat,dominance=float(np.mean(np.minimum(abs(hues.astype(float)-peak),180-abs(hues.astype(float)-peak))<=12)))
def support_at(maps,hit,width,height):
    f=hit['frame_id'];x=int(round(hit['x']*320/width));y=int(round(hit['y']*180/height));values=[]
    for t in (f-1,f,f+1):
        m=maps.get(t)
        if m is None:return dict(veto=False,reason='uncertain_template',support=values)
        patch=m['mask'][max(0,y-5):min(180,y+6),max(0,x-5):min(320,x+6)]
        if not patch.size:return dict(veto=False,reason='outside_map',support=values)
        values.append(dict(frame=t,fraction=float(patch.mean()),hue=m['hue'],dominance=m['dominance']))
    return dict(veto=max(v['fraction'] for v in values)<.10,reason='color_support',support=values)
class Solution(FrozenSolution):
    def prepare(self,context):
        super().prepare(context);self.color_dir=Path(context.output_dir)/'color_evidence';self.color_dir.mkdir(exist_ok=True)
    def reset(self,meta):super().reset(meta);self.color_maps={}
    def process_frame(self,rgb,meta):
        result=super().process_frame(rgb,meta);f=meta['frame_id']
        self.color_maps[f]=color_map(rgb,self.table_boxes.get(f));return result
    def finish_video(self,meta):
        baseline=super().finish_video(meta)
        points,accepted,_=fuse(dict(self.recovery.points),self.candidates,meta.width,meta.height,True)
        combined,changes=transform([h.to_dict() for h in baseline],points,accepted,self.table_boxes,'combined')
        unique={(h.frame_id,h.x,h.y):h.to_dict() for h in baseline}
        unique.update({(h['frame_id'],h['x'],h['y']):h for h in combined})
        rows=[dict(hit=h,**support_at(self.color_maps,h,meta.width,meta.height)) for h in unique.values()]
        (self.color_dir/(meta.video_id+'.json')).write_text(json.dumps(rows,indent=2))
        return baseline

"""Independent hypotheses over trajectories; no video IDs or labels."""
import math
def conflicts(a,b):
    return abs(a['frame_id']-b['frame_id'])<7 and math.hypot(a['x']-b['x'],a['y']-b['y'])<=100
def inside(h,boxes):
    b=boxes.get(h['frame_id'])
    return b is None or b[0]<=h['x']<=b[2] and b[1]<=h['y']<=b[3]
def horizontal_acceleration(h,points):
    f=h['frame_id'];ps=[points.get(i) for i in (f-3,f-1,f+1,f+3)]
    if any(p is None or p[2]<.8 for p in ps):return False
    a,b,c,d=ps;vx0=(b[0]-a[0])/2;vx1=(d[0]-c[0])/2
    vy0=(b[1]-a[1])/2;vy1=(d[1]-c[1])/2
    return vx0*vx1>0 and abs(vx1)>2*abs(vx0) and abs(vx1-vx0)>max(12,1.5*abs(vy1-vy0))
def micro_peak(h,points):
    f=h['frame_id'];ps=[points.get(i) for i in range(f-2,f+3)]
    if any(p is None or p[2]<.8 for p in ps):return False
    a,b,c,d,e=ps
    return min(c[1]-a[1],c[1]-e[1])<6 and (2*c[1]-b[1]-d[1])<8
def short_rescue(hits,points,accepted,boxes):
    output=[dict(h) for h in hits];added=[]
    for f in sorted(accepted):
        ps=[points.get(i) for i in (f-1,f,f+1)]
        if any(p is None or p[2]<.8 for p in ps):continue
        a,b,c=ps;fall=b[1]-a[1];rise=b[1]-c[1];impulse=fall+rise
        dx0=b[0]-a[0];dx1=c[0]-b[0]
        if fall<.5 or rise<.5 or impulse<10 or abs(dx1-dx0)>impulse:continue
        if dx0*dx1<0 and min(abs(dx0),abs(dx1))>2:continue
        if max(math.dist(a[:2],b[:2]),math.dist(b[:2],c[:2]))>220:continue
        h=dict(frame_id=f,x=b[0],y=b[1])
        if not inside(h,boxes) or any(conflicts(h,old) for old in output):continue
        output.append(h);added.append(h)
    return output,added
def transform(hits,points,accepted,boxes,mode):
    output=[dict(h) for h in hits];added=[];removed=[]
    if mode in ('short','short_horizontal','combined'):output,added=short_rescue(output,points,accepted,boxes)
    for name,predicate in [('horizontal',horizontal_acceleration),('micro',micro_peak)]:
        if mode not in (name,'combined') and not (name=='horizontal' and mode=='short_horizontal'):continue
        keep=[]
        for h in output:
            if predicate(h,points):removed.append(dict(h,reason=name))
            else:keep.append(h)
        output=keep
    return output,dict(added=added,removed=removed)

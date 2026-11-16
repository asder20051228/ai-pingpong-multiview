import json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/aiinfer/pingpang_training/official_eval')
DATA = ROOT / 'data/public_data'

def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def match(ps, rs):
    edges = [[i for i,r in enumerate(rs) if abs(p['frame_id']-r['frame_id'])<=1 and math.hypot(p['x']-r['x'],p['y']-r['y'])<=50] for p in ps]
    assigned = {}
    def aug(pi, seen):
        for gi in edges[pi]:
            if gi in seen: continue
            seen.add(gi)
            if gi not in assigned or aug(assigned[gi], seen):
                assigned[gi] = pi
                return True
        return False
    for i in range(len(ps)): aug(i, set())
    return assigned

def evaluate(path, policy='lenient'):
    preds, gt = defaultdict(list), defaultdict(list)
    for p in read_rows(path): preds[p['video_id']].append(p)
    for g in read_rows(DATA/'gt_public.jsonl'): gt[g['video_id']].append(g)
    counts = defaultdict(lambda: [0,0,0]); details = {}
    for video in json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))['videos']:
        vid=video['video_id']; ps=preds[vid]; valid=[g for g in gt[vid] if g['valid']]
        ignored=[g for g in gt[vid] if not g['valid']]
        if policy=='lenient':
            drop=set(match(ps,ignored).values()); ps=[p for i,p in enumerate(ps) if i not in drop]
        pairs=match(ps, valid); tp=len(pairs); fp=len(ps)-tp; fn=len(valid)-tp
        c=counts[video['angle']]
        for i,n in enumerate((tp,fp,fn)): c[i]+=n
        details[vid]={'counts':[tp,fp,fn], 'matched':[{'gt':valid[g], 'prediction':ps[p]} for g,p in pairs.items()], 'missed':[g for i,g in enumerate(valid) if i not in pairs], 'false_positives':[p for i,p in enumerate(ps) if i not in pairs.values()]}
    f1={a:2*c[0]/(2*c[0]+c[1]+c[2]) for a,c in counts.items()}
    return {'policy':policy,'counts':dict(counts),'f1':f1,'points_of_75':25*sum(f1.values()),'details':details}

if __name__=='__main__':
    import sys
    for p in sys.argv[1:]:
        r={policy:evaluate(p,policy) for policy in ('strict','lenient')}
        Path(p).with_name('scores.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
        print(p, {k:{q:v[q] for q in ('counts','points_of_75')} for k,v in r.items()},flush=True)

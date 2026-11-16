"""Recheck local public-set scores; this is not the hidden official evaluator."""
from pathlib import Path
import argparse
import importlib.util
import json

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('predictions',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
spec=importlib.util.spec_from_file_location('scorer',ROOT/'evidence/frozen_scoring.py')
scorer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)
scorer.DATA=ROOT/'participant/pingpang/public_data'
result={policy:scorer.evaluate(args.predictions,policy) for policy in ('strict','lenient')}
if args.output:
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({policy:{key:value[key] for key in ('counts','points_of_75')}
                  for policy,value in result.items()},ensure_ascii=False,indent=2))

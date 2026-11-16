#!/usr/bin/env bash
set -euo pipefail
OUT=/participant/output
PY=/opt/conda/envs/conda_env/bin/python3
mkdir -p "$OUT/selfcheck" /tmp/public_input
for task in pingpang basketball; do
    ln -s "/participant/$task/public_data" "/tmp/public_input/$task"
    cd "/participant/$task"
    "$PY" integrity.py --verify
    "$PY" scripts/check_env.py 2>&1 | tee "$OUT/selfcheck/${task}_environment.log"
    "$PY" scripts/check_solution.py --input public_data --solution participant.solution:Solution --decoder gpu 2>&1 | tee "$OUT/selfcheck/${task}_solution.log"
    cp -r scripts/out_check_solution "$OUT/selfcheck/$task"
    mkdir -p "$OUT/$task"
    if [ -d scripts/out_check_solution/trt_cache ]; then
        cp -r scripts/out_check_solution/trt_cache "$OUT/$task/"
    fi
done
/participant/run_all.sh /tmp/public_input "$OUT" 2>&1 | tee "$OUT/joint_console.log"

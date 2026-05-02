#!/bin/bash
set -euo pipefail

export ATEN_CPU_CAPABILITY=default
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p /logs/verifier
echo "0" > /logs/verifier/reward.txt

python /tests/run_verifier.py 2>&1 | tee /logs/verifier/pytest.log || true

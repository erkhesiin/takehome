#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
echo "0" > /logs/verifier/reward.txt

python /tests/run_verifier.py > /logs/verifier/pytest.log 2>&1 || true

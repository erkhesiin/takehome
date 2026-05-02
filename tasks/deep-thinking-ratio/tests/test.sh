#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier

python /tests/run_verifier.py > /logs/verifier/pytest.log 2>&1

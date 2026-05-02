#!/bin/bash
set -u

mkdir -p /logs/verifier

pytest /tests/test_dtr.py -q | tee /logs/verifier/pytest.log

passed=$(grep -Eo '[0-9]+ passed' /logs/verifier/pytest.log | tail -n 1 | awk '{print $1}')

passed=${passed:-0}

total=7
reward=$(python -c "print(${passed} / ${total})")
echo "${reward}" > /logs/verifier/reward.txt

exit 0

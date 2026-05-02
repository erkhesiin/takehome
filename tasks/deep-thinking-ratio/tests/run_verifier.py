from pathlib import Path

import pytest


TOTAL_CASES = 7
REWARD_PATH = Path("/logs/verifier/reward.txt")


class PassCounter:
    def __init__(self) -> None:
        self.passed = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" and report.passed:
            self.passed += 1


def main() -> int:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    counter = PassCounter()

    try:
        pytest.main(["/tests/test_dtr.py", "-q", "-rA"], plugins=[counter])
    finally:
        reward = counter.passed / TOTAL_CASES
        REWARD_PATH.write_text(f"{reward}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

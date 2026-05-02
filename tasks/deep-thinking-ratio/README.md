# takehome/deep-thinking-ratio

This Harbor task asks an agent to implement the Deep-Thinking Ratio algorithm
from the included DTR paper. The agent prompt is in `instruction.md`; the
expected deliverable is a single importable file at `/solution/dtr.py` defining
`compute_dtr`.

## Environment

The task uses `python:3.11-slim` with `torch==2.3.1` and `pytest==8.4.1`
installed at image build time. Runtime internet is disabled. The task is CPU
only with 2 CPUs, 2048 MB RAM, 10240 MB storage, a 600 second agent timeout,
and a 120 second verifier timeout.

## Verifier

The verifier is deterministic pytest. It generates synthetic hidden states and
unembedding matrices from fixed Torch seeds, calls the submitted implementation,
and compares it with a reference DTR implementation.

| Dimension | Type | What it measures |
| --- | --- | --- |
| Accuracy | Programmatic | Fraction of seven DTR cases matching the reference within 0.01 |

Reward is written to `/logs/verifier/reward.txt` as `passed_cases / 7`.

## Layout

```text
tasks/deep-thinking-ratio/
├── DTR_Paper.pdf
├── instruction.md
├── task.toml
├── environment/
│   ├── Dockerfile
│   └── DTR_Paper.pdf
├── solution/
│   ├── dtr.py
│   └── solve.sh
└── tests/
    ├── test.sh
    ├── run_verifier.py
    ├── test_dtr.py
    ├── test_cases.py
    └── dtr_reference.py
```

## Running

```bash
harbor run -p tasks/deep-thinking-ratio --agent oracle
harbor run -p tasks/deep-thinking-ratio --agent codex --model openai/gpt-5.5
```

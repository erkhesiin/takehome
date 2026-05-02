# Write Up

Kevin Tamir (amarin)

## Paper

TODO: Explain why "Closed-form feedback-free learning with forward projection" was chosen and summarize the target-potential and closed-form fitting equations.

I had a list of papers I was thinking of doing and since I was given the time of 
~2 days, I decided to take advantage of it. I tested each paper, starting with the easiest, but newest, to see if a frontier model would struggle. It did not. [Forward Projection](https://arxiv.org/abs/2501.16476), on the other hand, had that novelty to it that could possible make a frontier model struggle to implement a AI/ML research paper, and it did to a better extent than any of the other papers I had planned. In the field of training AI to do research on AI, implementing skills from novel ideas is a great idea and something that I think frontier labs have done for a long while now across tons of different topics. This can be best seen in the various different instances where people used 5.5-Pro to do math research and it had genuinely good insights. However, it struggles on doing purely novel work such as developing something from first principles and to my understanding, this is where Hillclimb comes in. It will help AI develop these skills through various means at which I don't remember well enough to write about. In any case, giving an AI the task of implementing a novel paper means it doesn't have as much information on it in its training data so it has to come up with its own novel ways to implement it. To do it with a paper that challenges backprop, something that's been thought of to be **the way** models learn, also would give a frontier model some amount of trouble as it goes against an assumed common consensus in their training data. 

## Capability

TODO: Explain the capability being tested: translating a recent ML algorithm into numerically stable PyTorch without relying on gradient descent or memorized reference code.

## Agent Task

TODO: Describe what the agent is asked to create at `/solution/forward_projection.py`.

## Environment

TODO: Describe the Docker image, available libraries, CPU-only constraints, and source paper artifact.

## Verifier And Reward

TODO: Describe the deterministic pytest verifier and the partial-credit reward calculation.

## Design Decisions And Tradeoffs

TODO: Explain why the task focuses on the core FP equations rather than full biomedical datasets or full model reproduction.

I was originally designing this task for the paper ["Think Deep, Not Just Long"](https://arxiv.org/abs/2602.13517); However, I looked back on the details of the take-home assignment and appears we would want to be working "towards tasks that challenge frontier models without artificially skewing rewards," and this makes sense. What's the point of doing all of this when all the agent is doing is creating a simple function. As one can guess, GPT-5.5 and Claude Opus 4.7 would have no issue implementing something like DTR, and during initial test, it indeed did have a mean score of 1.0. I decided to pivot into the [Mamba paper](https://arxiv.org/abs/2312.00752) for a brief period before realizing something with a publish date in 2023 is ancient and GPT-5.5/Opus 4.7 *probably* has already been trained on it and has the same issue of DTR of being too easy for the model to solve, which Opus 4.7 did with a mean score of 0.986. These scores were far too high to actually signal struggle from the side of the model and stems from me just choosing papers with the wrong mindset. From there, I pivoted into the Forward Projection Paper. 

By the time I finished, I did see this constant pivoting sort of as a way to show how quickly I can adapt to new conditions, something I think is crucial in the research space as no research paper ever had a linear experimental timeline. In any case, doing the final pivot into the FP paper proved to be the correct move because Opus 4.7 had a bit of trouble, having a mean score of 0.800. GPT-5.5 on the other hand had a score of 0.933.

## Sample Rollouts

I ran the required 10 rollouts for both Claude Code Opus 4.7 high and Codex GPT-5.5 xhigh on the Forward Projection task. The results were meaningfully different from the earlier DTR and Mamba attempts: the task was not impossible, but it also did not collapse into every rollout getting a perfect score.

| Agent | Model / effort | Trials | Mean reward | Full-credit trials | Notable issue | Wall-clock |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Claude Code | `anthropic/claude-opus-4-7`, high | 10 | 0.800 | 4/10 | 6 trials missed the prediction replay behavior | 11m 02s |
| Codex | `gpt-5.5`, xhigh | 10 | 0.933 | 9/10 | 1 trial failed most fit cases; 1 full-credit trial also hit an agent-timeout flag | 35m 58s |

Claude was fast and usually implemented the closed-form fitting path correctly. Its partial-credit failures were very consistent: six trials scored `0.6667` because they passed the eight fitting tests but failed all four `forward_projection_predict` tests. The failed assertion was that `predict activations` had the wrong length, which means those implementations did not return `[A_0, A_1, ..., A_L]` during replay. This is exactly the kind of useful signal I wanted: the model understood the main equation, but missed an API/detail requirement that matters for composing the learned layers.

| Claude trial | Reward | Total duration | Main result |
| --- | ---: | ---: | --- |
| `5yGY6gL` | 1.000 | 1m 59s | full pass |
| `H3YGako` | 0.667 | 1m 45s | failed all prediction replay cases |
| `Pihgae9` | 0.667 | 2m 01s | failed all prediction replay cases |
| `Vt3AZ9E` | 1.000 | 1m 36s | full pass |
| `Xsv9WDb` | 0.667 | 1m 48s | failed all prediction replay cases |
| `aMKQsMU` | 0.667 | 2m 07s | failed all prediction replay cases |
| `gZY3fa3` | 0.667 | 1m 44s | failed all prediction replay cases |
| `k4sPMXA` | 1.000 | 3m 25s | full pass |
| `kEpprJC` | 1.000 | 2m 00s | full pass |
| `zmCkFsQ` | 0.667 | 2m 08s | failed all prediction replay cases |

Codex spent longer per rollout and produced more verbose work, but usually landed on a complete implementation. Nine of the ten trials received reward `1.0`. The one low-scoring trial, `Sj3QVfq`, scored `0.3333`: it passed the prediction tests but failed all eight fitting tests, with large numerical mismatches in the fitted weights/membranes. One other trial, `KqLdbDL`, recorded an `AgentTimeoutError` at the Harbor level even though the artifact that was present verified at reward `1.0`; I am treating that as an operational rollout issue and keeping it visible rather than hiding it.

| Codex trial | Reward | Total duration | Main result |
| --- | ---: | ---: | --- |
| `6fH84MW` | 1.000 | 4m 40s | full pass |
| `HJ7sTej` | 1.000 | 5m 04s | full pass |
| `K6K2pnx` | 1.000 | 9m 09s | full pass |
| `KqLdbDL` | 1.000 | 10m 29s | full pass artifact; agent timeout flag |
| `Sj3QVfq` | 0.333 | 4m 24s | failed all fitting cases |
| `h6Amnmm` | 1.000 | 6m 04s | full pass |
| `hu4QvHn` | 1.000 | 4m 26s | full pass |
| `t6NKAu5` | 1.000 | 5m 18s | full pass |
| `uNnLjRQ` | 1.000 | 6m 51s | full pass |
| `zuYcubz` | 1.000 | 9m 32s | full pass |

These rollouts suggest the task has a healthier reward profile than the earlier attempts. DTR gave no learning signal because both frontier models saturated it. Mamba was also too culturally cached as an implementation target. Forward Projection still allows strong models to succeed, but the failures are structured and interpretable: missing replay semantics, incorrect closed-form fitting, and occasional agent/runtime fragility.

## Possible Improvements

TODO: Note follow-up versions, such as streaming sufficient-statistics checks, local label decoding, or small end-to-end classification experiments.

# Deduction

To get full points for deduction, we need to achieve 86% accuracy (human-level) on LogiQA 2.0 (Liu et al., 2023) benchmark. As I pointed out in the checkpoint paper, the results might not be representative of the model's actual reasoning capabilites due to data contamination - the benchmark might have appeared in the model's training data. The authors of [A Definition of AGI](https://www.agidefinition.ai/) state that their implementation of the benchmark was subject to memorization and robustness checks, but no actual details are revealed how they did it. As such, I will run the benchmark unchanged, then with some alterations, probing and finally I will use the JustLogic  (M.  K.  Chen  et  al., 2025) benchmark, which is built with the contamination problem in mind - it uses sythetic items to isolate the deduction abilites of the model, no memorization is possible from brior exposure.

These benchmarks are usually ran through API, but due to reasons outlined by [Aaron](theory_of_mind\README.md), I ran them through Claude Code. The LogiQA2.0 repository doesn't have scripts to run the benchmarks with, so with the help of Claude I made a Python script that feeds the questions from the dataset to Claude Code one by one. 

**Baseline (300 questions)**

| Category | Accuracy |
|---|---|
| Categorical reasoning | 90.4% |
| Sufficient conditional reasoning | 91.1% |
| Necessary conditional reasoning | 95.2% |
| Disjunctive reasoning | 92.6% |
| Conjunctive reasoning | 90.1% |
| **Overall** | **91.3%** |
| **Macro average** | **91.9%** |

**Combined variant (100 questions)**
*Combined variant: each question has its person names replaced with invented ones, two irrelevant sentences added to the passage, and the answer options shuffled. The underlying logic and correct answer are unchanged.*
| Category | Accuracy |
|---|---|
| Categorical reasoning | 91.3% |
| Sufficient conditional reasoning | 89.6% |
| Necessary conditional reasoning | 93.2% |
| Disjunctive reasoning | 84.6% |
| Conjunctive reasoning | 88.6% |
| **Overall** | **90.0%** |
| **Macro average** | **89.5%** |

**Memorization checks**

For memorization checks, I probed whether Claude could reproduce LogiQA 2.0 test items instead of solving them. In a completion probe, Claude received the first half of a passage and was asked to continue it. In an option-recall probe, it received the passage and question without the answer options and was asked to write the four options. Each probe was run in two ways: once with no mention of where the text came from, and once telling Claude the text was from the LogiQA 2.0 test set and asking it to reproduce it exactly. 

| Probe | Mode | n | Mean ROUGE-L | ROUGE-L > 0.5 | Shared run ≥ 8 words |
|---|---|---|---|---|---|
| Completion | Plain | 20 | 0.281 | 2 | 1 |
| Completion | Guided | 20 | 0.548 | 8 | 9 |
| Options | Plain | 20 | 0.433 | 7 | 8 |
| Options | Guided | 20 | 0.485 | 9 | 8 |

The model has clearly seen at least some of the data. Looking at the result in *probes_results.jsonl*, we see that Claude manages to perfectly or near perfectly reconstract many passages, options. Looking at the models output, where we see reasoning, it says things like "Reconstructed from the original five-option LSAT question; I can't verify which option LogiQA 2.0 dropped or its exact wording." This implies heavily that the model saw LogiQA during training, which is the predecessor to LogiQA 2.0

# What can we conclude from LogiQA 2.0 results

Looking naively at the results, we could deduce that Claude Opus 5.5 has human level resoning. But obviously we cannot say it for certain. The model is familiar with some of the data, as we proved with probing. This is why we also JustLogic.

# Justlogic
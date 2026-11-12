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


**Memorization checks**

For memorization checks, I probed whether Claude could reproduce LogiQA 2.0 test items instead of solving them. In a completion probe, Claude received the first half of a passage and was asked to continue it. In an option-recall probe, it received the passage and question without the answer options and was asked to write the four options. Each probe was run in two ways: once with no mention of where the text came from, and once telling Claude the text was from the LogiQA 2.0 test set and asking it to reproduce it exactly. 

**Memorization probes (100 questions)**

| Probe | Mode | n | Mean ROUGE-L | ROUGE-L > 0.5 | Shared run ≥ 8 words |
|---|---|---|---|---|---|
| Completion | Plain | 100 | 0.292 | 11 | 9 |
| Completion | Guided | 100 | 0.518 | 42 | 42 |
| Options | Plain | 100 | 0.378 | 23 | 27 |
| Options | Guided | 100 | 0.497 | 48 | 41 |

The model has clearly seen at least some of the data. Looking at the result in *probes_results.jsonl*, we see that Claude manages to perfectly or near perfectly reconstract many passages, options. Looking at the models output, where we see reasoning, it says things like "Reconstructed from the original five-option LSAT question; I can't verify which option LogiQA 2.0 dropped or its exact wording." It seems like the model has seen questions in LogiQA 2.0 from other contexts. Lets try another test. 

**Passage removed (100 questions)**

| Category | Accuracy |
|---|---|
| Categorical reasoning | 71.0% |
| Sufficient conditional reasoning | 74.0% |
| Necessary conditional reasoning | 72.7% |
| Disjunctive reasoning | 69.2% |
| Conjunctive reasoning | 73.4% |
| **Overall** | **73.0%** |
| **Macro average** | **72.1%** |

*Passage removed: each question was shown with only the question and the four answer options; the passage was replaced by a note saying it had been removed. Chance level is 25%.*
 
 These results support the idea that Claude has seen this data before. There is a possibility of a model being able to pick the correct option solely on the basis of it sounding the most reasonable, but 73.0% is simply way too high.

# Can we battle against memorization?

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

Accuracy on the combined variant (90%) was close to the baseline, so Claude is robust to surface changes such as renamed entities, irrelevant sentences and shuffled options, and it does not depend on remembering answer positions. However, this variant leaves each item's content and logic largely intact, so a model that recognizes an item can still choose the same answer by content

**Shuffled options + correct answer replaced with "None of the other options is correct" (100 questions)**

| Category | Accuracy |
|---|---|
| Categorical reasoning | 71.0% |
| Sufficient conditional reasoning | 64.6% |
| Necessary conditional reasoning | 61.4% |
| Disjunctive reasoning | 50.0% |
| Conjunctive reasoning | 58.2% |
| **Overall** | **65.0%** |
| **Macro average** | **61.0%** |

 We did some heavy modification; they force the model to reason about each option and rule them out one by one, although being familiar with the dataset could mean that the model could recognize the correct answer is missing, and reach the “None of the above” trivially. Still, the accuracy is very low compared to the baseline, and to be fair, this is a harder benchmark now.  So the result is probably quite fair and hopefully isolates deduction, but we don't know how humans would to on modifications like this.

# What can we conclude from LogiQA 2.0 results

Looking naively at the results, we could deduce that Claude Opus 5.5 has human level resoning. But obviously he model is familiar with the data, as we proved with very excplicit memorization tests, so we can not rely on this benchmark at all. The robustness checks we did may have helped, and maybe the results from that test really do isolate reasoning, but we cannot say that for certain.

In the end, I ran just 300 randomly chosen samples on the unchanged data, and the first 100 of those samples very modified for robustness and memorization checks. The reason why I didn't use the full test set data (about 1400 questions) is precisely due to the fear of data contamination, which we proved; running the whole benchmark would be pointless since we can already see we shouldn't trust its results.

# Justlogic


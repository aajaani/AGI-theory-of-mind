For Theory of Mind, there are 2 benchmarks where Claude Opus 5.5 has to beat the human scores (85.4% on ToMBench and 87.5% on FANToM). Namely ToMBench (Z. Chen et al., 2024) and FANToM (Kim et al., 2023).

I've also chosen a third benchmark to test which is OpenToM (Xu et al., 2024) which covers some of the aspects of Theory of Mind not covered by the other two benchmarks, although the AGI definitions paper only talks about the first two benchmarks.

# TOMBench

### Data

Every question carries three labels. First there are the 6 categories, which describe what kind of mental state is involved (Emotion, Desire, Intention, so on).

Then there are the 31 abilities, which mean the specific skill within that category, for example "Location false beliefs" or "White lies".

Then there are the tasks, where there are 8, this means the test format the question came from, for example Unexpected Outcome Test, False Belief Task, so on.

So the Category and ability form a hierarchy, where each ability belongs to one category. Tasks are independent of both, it's just a style of test, one task can test several abilities.

In their public repository they have 8 task files and 12 separate ability files, that cover abilities which the 8 tasks do not cover. However we do not need to run this benchmark on the 12 separate ability files since they are not included in the human baseline average (85.4%) that the AGI definitions paper uses.

So to sum up, to run a test run of the benchmark which covers all of the tests, we have to sample from every combination of task and ability (23 in total), not just from each category or each task. Taking 50 questions from each task, split between its abilities, gives a set of 400 questions that still includes every task, every ability the tasks cover and every category. For the test runs below I used a smaller set of 24 per task (192 questions) out of those.

### Evaluation

#### Language

TOMBench was run on 10 models with both Chinese and English. It found that the best model GPT-4-1106 was 1.3 % better on Chinese. I will run the tests on English, since all the other tests for theory of mind should be in English as well, and it offers better comparability. For the human benchmark, it was only measured with Chinese speakers. They justify this by citing a study (Bradford et al., 2018) which claims that native english and native chinese speakers perform about the same on theory of mind tasks.

#### Code

The repository contains all of the 2860 handwritten questions and the python code to run a model on them and score it. The given repository has some issues though. Just running it plainly will crash, there is a bug where the yes/no questions in the Faux-pas and Strange Story files crash the evaluation code. The python code in the repository also is written for ChatGPT API, not claude. To fix those errors and adapt the code for Claude, I used Claude to help me.

#### How the questions were asked

All of the runs below used the 192 question set. Like in the ToMBench paper, every question was asked 5 times with the answer options in a different order each time and the most common answer was counted, so a model can't score well by always picking the same letter. The option orders were the same in every run, so when two runs are compared, the only difference is the thing being tested.

#### Memorization test

There is also the worry of contamination. This benchmark was created in 2024 but Opus 5.5 training cutoff is 2026. The AGI definitions paper gives a few ideas on trying to create a test set, for example by changing the questions slightly. I did it by having Claude swap out names of the characters and objects they interact with (objects only in the False Belief task) and comparing if the results differ.

I tried doing that and the results had no statistical difference with p=1. Both the original and the swapped version got 86.5%, 4 questions went from right to wrong and 4 went from wrong to right. 2 of those 8 were Scalar Implicature questions where nothing was even swapped, so most of that is probably just the normal randomness between runs. This does not necessarily mean that there's absolutely no memorization effect happening, but on the 192 question set i ran, it found no evidence of it happening either.

#### API vs Claude Code

The paper relied on API calls to run the prompts. We could do that, but it requires spending money to unlock the API key, it's tied to a different payment model compared to a normal subscription. Since all of us already had a regular claude subscription with access to claude code, our project's idea (mainly to save money), was to use claude code. To justify this, i ran another test, one with api calls and one with claude code. I made sure that they could not use tools or the internet and both had medium reasoning.

For Claude Code I ran it from an empty folder with all tools turned off and replaced its system prompt with the benchmark prompt, so it got the same instructions as the API. Claude Code also has an "advisor" tool (it lets the model ask another model for help) which stays on even when the other tools are off, so that had to be turned off separately. 

The API got 85.9% and Claude Code got 86.5%. They gave the same final answer on 180 of the 192 questions, 5 questions were only right with the API and 6 only with Claude Code (no statistical significance, p=1). The one real difference was that Claude Code used fewer output tokens, about 49 per answer compared to 78 with the API, so it reasons a bit less, but that didn't change the accuracy. The API run cost about 3.45 dollars

## Results

These are the results on the 192 question set, compared to the human scores from the ToMBench paper.

| Task                      | Claude API | Claude Code | Claude Code, swapped names | Human    |
| ------------------------- | ---------- | ----------- | -------------------------- | -------- |
| Unexpected Outcome Test   | 91.7       | 87.5        | 83.3                       | 89.3     |
| Scalar Implicature Test   | 62.5       | 62.5        | 62.5                       | 75.5     |
| Persuasion Story Task     | 66.7       | 66.7        | 66.7                       | 70.0     |
| False Belief Task         | 95.8       | 95.8        | 95.8                       | 86.8     |
| Ambiguous Story Task      | 87.5       | 95.8        | 95.8                       | 95.0     |
| Hinting Test              | 100.0      | 100.0       | 100.0                      | 97.1     |
| Strange Story Task        | 95.8       | 100.0       | 100.0                      | 89.2     |
| Faux-pas Recognition Test | 87.5       | 83.3        | 87.5                       | 80.4     |
| **Average**               | **85.9**   | **86.5**    | **86.5**                   | **85.4** |

Claude got 86.5% with Claude Code (95% confidence interval 82.3–90.6%), which is just above the human average of 85.4%. But since the interval goes well below 85.4%, with this many questions we can't say yet if it actually beats the human baseline.

Claude did better than humans on False Belief, Strange Story, Hinting and Faux-pas, and worse on Scalar Implicature and Persuasion Story, which are also the two tasks where humans scored the lowest. While looking at the wrong answers in the first test runs I also noticed that some of the questions look flawed. In the False Belief task some questions ask what a character expects to find "after" opening a box, when they have already seen what's inside (the Chinese original says the same). In the Scalar Implicature task some answer keys don't match simple arithmetic, for example a class with 50 students and 12 girls where the key says 35 boys. I still count these as wrong, since that's how the benchmark is scored.

Still to do:

- Run the full 400 questions, although I am leaning towards trying to run the full dataset, right now I did hit a limit with the test set above, although I also wasted a lot of the limit doing different things too, so I think it should be doable if I start early.

# FANToM

todo

# OpenToM

todo

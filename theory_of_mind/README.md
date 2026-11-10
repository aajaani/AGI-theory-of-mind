# TOMBench

Every question carries three labels. First there are the 6 categories, which describe what kind of mental state is involved (Emotion, Desire, Intention, so on).

Then there are the 31 abilities, which mean the specific skill within that category, for example "Location false beliefs" or "White lies".

Then there are the tasks, where there are 8, this means the test format the question came from, for example Unexpected Outcome Test, False Belief Task, so on.

So the Category and ability form a hierarchy, where each ability belongs to one category. Tasks are independent of both, it's just a style of test, one task can test several abilities.

In their public repository they have 8 task files and 12 separate ability files, that cover abilities which the 8 tasks do not cover. However we do not need to run this benchmark on the 12 separate ability files since they are not included in the human baseline average (85.4%) that the AGI definitions paper uses.

So to sum up, to run a test run of the benchmark which covers all of the tests, we have to sample from every combination of task and ability (23 in total), not just from each category or each task. Taking 50 questions from each task, split between its abilities, gives a set of 400 questions that still includes every task, every ability the tasks cover and every category. For the test runs I used a smaller set of 24 per task (192 questions) out of those.

TOMBench was run on 10 models, including GPT4, with both Chinese and English. It found that the best model GPT-4-1106 was 1.3 % better on chinese. Still though, for stronger models that difference should be smaller and I will run the tests on english, since all the other tests for theory of mind should be in english as well, and it offers better comparability. For the human benchmark, it was only measured with Chinese speakers. They justify this by citing a study (Bradford et al., 2018) which claims that native english and native chinese speakers perform about the same on theory of mind tasks.

The repository contains all of the 2860 handwritten questions and the python code to run a model on them and score it. The given repository has some issues though. Just running it plainly will crash, there is a bug where the yes/no questions in the Faux-pas and Strange Story files crash the evaluation code. The python code in the repository also is written for ChatGPT api, not claude.

There is also the worry of contamination. This benchmark was created in 2024 but Opus 5.5 training cutoff is 2026. The AGI definitions paper gives a few ideas on trying to create a test set, for example by changing the questions slightly. I did it by swapping out names of the characters and objects they interact with (objects only in the False Belief task) and comparing if the results differ.

I tried doing that and the results had no statistical difference with p=1. This does not necessarily mean that there's absolutely no memorization effect happening, but on the 192 question set i ran, it found no evidence of it happening either.

Also, the paper relied on API calls to run the prompts. We could do that, but it requires spending money to unlock the API key, it's tied to a different payment model compared to a normal subscription. Since all of us already had a regular claude subscription with access to claude code, our project's idea (mainly to save money), was to use claude code. To justify this, i ran another test, one with api calls and one with claude code. I made sure that they could not use tools or the internet and both had medium reasoning. The results similarly showed no statistical difference, with p=1. Also both of them were right on the cusp of the human benchmark, so i will run the full dataset on claude code in the future.

https://github.com/seacowx/OpenToM

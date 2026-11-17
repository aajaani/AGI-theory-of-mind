Updated 2026-11-17 12:29

| Benchmark | n | Correct | Accuracy | 95% CI | p (acc < 90%) | Unparsed |
|---|---|---|---|---|---|---|
| NP Trip | 100 | 99 | 99.0% | 94.6–99.8% | 1 | 0 |
| NP Meeting | 100 | 97 | 97.0% | 91.5–99.0% | 0.998 | 0 |
| NP Calendar | 100 | 92 | 92.0% | 85.0–95.9% | 0.794 | 0 |
| Blocksworld | 500 | 500 | 100.0% | 99.2–100.0% | 1 | 0 |
| Mystery BW | 500 | 498 | 99.6% | 98.6–99.9% | 1 | 0 |
| BW large | 110 | 110 | 100.0% | 96.6–100.0% | 1 | 0 |

Natural Plan mean of task accuracies (trip, meeting, calendar): 96.0%

Accuracy by problem size
- NP Trip by num_cities: 3: 13/13, 4: 13/13, 5: 13/13, 6: 13/13, 7: 12/12, 8: 12/12, 9: 12/12, 10: 11/12; logistic regression not estimable (all correct, or errors only at the largest sizes)
- NP Meeting by num_people: 1: 10/10, 2: 10/10, 3: 10/10, 4: 10/10, 5: 10/10, 6: 10/10, 7: 10/10, 8: 10/10, 9: 8/10, 10: 9/10; logistic b = -0.98 (SE 0.58), OR = 0.38, z = -1.68, p = 0.0932
- NP Calendar by num_people: 2: 47/50, 3: 8/10, 4: 10/10, 5: 8/10, 6: 10/10, 7: 9/10; logistic b = -0.04 (SE 0.20), OR = 0.96, z = -0.20, p = 0.838
- Blocksworld by plan_length: 2: 30/30, 4: 56/56, 6: 114/114, 8: 139/139, 10: 113/113, 12: 46/46, 14: 1/1, 16: 1/1; logistic regression not estimable (all correct, or errors only at the largest sizes)
- Mystery BW by plan_length: 2: 30/30, 4: 56/56, 6: 114/114, 8: 139/139, 10: 112/113, 12: 45/46, 14: 1/1, 16: 1/1; logistic b = -0.58 (SE 0.32), OR = 0.56, z = -1.81, p = 0.0708
- BW large by n_blocks: 6: 1/1, 7: 1/1, 8: 7/7, 9: 14/14, 10: 13/13, 11: 12/12, 12: 13/13, 13: 18/18, 14: 21/21, 15: 10/10; logistic regression not estimable (all correct, or errors only at the largest sizes)

Paired comparison on 500 shared instances: Blocksworld 100.0% vs Mystery 99.6%; discordant pairs 2 (only BW correct) / 0 (only Mystery correct); exact McNemar p = 0.5

Mean output tokens / seconds per item
- NP Trip: 597 tokens, 5 s
- NP Meeting: 1012 tokens, 9 s
- NP Calendar: 144 tokens, 2 s
- Blocksworld: 201 tokens, 3 s
- Mystery BW: 823 tokens, 8 s
- BW large: 870 tokens, 8 s

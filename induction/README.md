# Induction
Induction is the capacity to determine general rules and principles underlying a problem or phenomenon, based on specific observations. To test for induction, I have used the ARC-AGI-2 Public Evaluation dataset containing 120 tasks. Each task contains "train" and "test" data: demonstration and test input/output pairs constiting of grids, which each cell mapped to a specific color. The test can be considered passed if the taker picks the right grid size and the correct colors for each cell in the grid, corresponding to the right answer based on the input. ARC-AGI-2 Human testing dataset was used to establish a human baseline. Thus, full points for induction can be obtained if the model performs better than the 90th percentile. 

Hendrycks et al. (2025) used a private dataset when benchmarking for induction. The problem with using the public ARC-AGI-2 dataset is contamination, as it is possible that Anthropic's Claude Opus 5.5 model might have been trained on the data already. To try and see if the model recalls answers or applies inductive reasoning, I have applied transformations in order to disguise the original data from the evaluation dataset: randomly remapping the colors and either flipping the grid either horizontally/vertically, or applying a 180-degree rotation. 

To run the benchmark, for reasons already outlined in the project, I have used Claude Code instead of Claude API. To this end, I had Claude adapt a Python script for running the test in the ARC-AGI Benchmarking repository through Claude Code instead of API.

## Disguised data

This is an example of what data looks like before and after the transformations. To generate this output I have used the visualization tool provided in the ARC-AGI Benchmarking repo. 

<img width="498" height="479" alt="Screenshot 2026-11-16 at 19 02 50" src="https://github.com/user-attachments/assets/fcd9f3da-e8d3-4030-92ed-a62219b3f0b9" />
<img width="496" height="482" alt="Screenshot 2026-11-16 at 19 03 13" src="https://github.com/user-attachments/assets/067c097e-8227-414c-b5bc-3f7040ca9d18" />


The transformations do not alter the grid dimensions, nor do they change the nature of the problem at hand. I have kept track of the transformations applied to each task so as to observe their effect on the results.

## Results
### Original & disguised

The benchmark was run using 120 tasks from the ARC-AGI-2 Public Evaluation dataset, containing a total of 167 test pairs.

| Data      | Accuracy (task-level) | 95% CI      | Avg. tokens per task |
|-----------|----------------------:|------------:|---------------------:|
| Original  | 80.7%                 | 74.4–86.7   | 17076               |
| Disguised | 83.6%                 | 77.6–89.2   | 17637               |


| Disguise                 | n   | Original | Disguised | Gap  |
|--------------------------|----:|---------:|----------:|-----:|
| Background 0 kept        | 17  | 91.2%    | 91.2%     | +0.0 |
| Background 0 remapped    | 103 | 79.0%    | 82.4%     | −3.4 |
| Flip: mirror horizontal  | 35  | 88.1%    | 85.2%     | +2.9 |
| Flip: mirror vertical    | 36  | 65.3%    | 73.6%     | −8.3 |
| Flip: rotate 180°        | 49  | 86.7%    | 89.8%     | −3.1 |

Using one attempt at passing the test for purposes of subscription usage (pass1), the model scored 80.7% (95% confidence interval 74.4-86.7%) on the original tasks and 83.6% (77.6-89.2%) on the disguised tasks. The difference of −2.9 points (95% CI −8.8 to +2.9) is not statistically significant (p = 0.345). This can point to the model's capacity to infer rules from the pairs without recalling previous answers. The upper bound of the interval indicates that any advantage from memorisation is at most about 3 points, so the effect is quite small.

### Comparing to human baseline

131 out of the 167 test outputs were correct for the model running on pass1. The human comparison uses a per-test-input score (78.9% on the 161 inputs with human data) rather than ARC's per-task score (80.7%), because human results are recorded per test input.

| Human rule | Human avg | Model, same pairs | Participants | Model percentile | Humans at 100% |
|------------|----------:|------------------:|-------------:|-----------------:|---------------:|
| pass1      | 41.2%     | 78.9%             | 187          | 88.2%            | 4.8%           |
| pass2      | 56.0%     | 78.9%             | 187          | 75.7%            | 9.6%           |
| any        | 65.7%     | 78.9%             | 187          | 61.5%            | 24.6%          |

Using both pass1 and pass2 as human testing baseline, the model running on pass1 scores below the 90th percentile, but above the human average. That would imply **a 20% increase of the total On-the-Spot reasoning score**. Running the benchmark on pass2 would allow for a direct comparison to the pass2 human rule.

The model can be familiar with the data, since ARC-AGI is a widely used benchmark for AI models. The second iteration of tests, ARC-AGI-2, are already over a year old and have been run on most frontier models with private testing datasets; ARC Prize Foundation's testing of Claude Opus 5.5 Medium on a semi-private dataset yielded a score of 87.5% on pass2. Next I would like to further test the model for induction using benchmarks from the survey by K. Chen et al. (2026).

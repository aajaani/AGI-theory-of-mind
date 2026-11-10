import argparse
import collections
import random

from tombench_data import TASK_FILES, load_all, read_questions, write_questions


def allocate(sizes, total):
    # Proportional allocation with largest remainders, so every ability in a
    # task gets its share of the task's questions.
    n = sum(sizes.values())
    exact = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (exact[k] - alloc[k], k), reverse=True)[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a fixed number of questions per ToMBench task, stratified by ability.")
    parser.add_argument("--per_task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--within", type=str, default="", help="sample from this question file instead of the full data")
    args = parser.parse_args()

    output = args.output or f"subsets/tombench_{args.per_task}pertask_seed{args.seed}.jsonl"
    rng = random.Random(args.seed)
    records = read_questions(args.within) if args.within else load_all()

    picked = []
    for task in TASK_FILES:
        groups = collections.defaultdict(list)
        for r in records:
            if r["file"] == task:
                groups[r["ability"]].append(r)
        alloc = allocate({a: len(g) for a, g in groups.items()}, args.per_task)
        print(task)
        for ability in sorted(groups):
            chosen = rng.sample(groups[ability], alloc[ability])
            picked.extend(chosen)
            print(f"  {alloc[ability]:3d} / {len(groups[ability]):3d}  {ability}")

    picked.sort(key=lambda r: (TASK_FILES.index(r["file"]), r["row"]))
    for r in picked:
        r["variant"] = "original"
    write_questions(output, picked)
    print(f"wrote {len(picked)} questions to {output}")

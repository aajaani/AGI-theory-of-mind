"""
Build the question files in subsets/ from the two public benchmarks.

Needs the benchmark repositories in external/ (not committed, see README):
    git clone --depth 1 https://github.com/google-deepmind/natural-plan external/natural-plan
    git clone --depth 1 https://github.com/karthikv792/LLMs-Planning external/LLMs-Planning

Usage:
    python prepare_data.py

Writes one JSON object per line to
    subsets/natural_plan_trip.jsonl        100 items, stratified by number of cities (3-10)
    subsets/natural_plan_meeting.jsonl     100 items, stratified by number of people (1-10)
    subsets/natural_plan_calendar.jsonl    100 items, stratified by (people, days)
    subsets/blocksworld.jsonl              all PlanBench Blocksworld task 1 instances
    subsets/mystery_blocksworld.jsonl      the same instances in the obfuscated domain
    subsets/blocksworld_hard.jsonl         110 larger instances (6-15 blocks), same prompt format

Items are ordered so that the first N of a file are themselves a balanced random
sample (round-robin over the strata), so a run with --limit N is a fair pilot.
Blocksworld and Mystery Blocksworld use the same instance order, so the first N
of both files are the same problems and can be compared pairwise.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
NP_DIR = HERE / "external" / "natural-plan" / "data"
PB_DIR = HERE / "external" / "LLMs-Planning" / "plan-bench"
OUT = HERE / "subsets"
SEED = 42
N_PER_TASK = 100

NP_SYSTEM = (
    "Solve the last TASK. Give your final answer in exactly the same format as the "
    "SOLUTION of the examples, and give only one plan."
)
BW_SYSTEM = (
    "Complete the plan for the last [STATEMENT]. Write the plan in the same format as the "
    "example: one action per line, and end it with [PLAN END]."
)

# PlanBench object letters -> names used in the prompts (configs/blocksworld.yaml)
BW_COLORS = {"a": "red", "b": "blue", "c": "orange", "d": "yellow", "e": "white", "f": "magenta",
             "g": "black", "h": "cyan", "i": "green", "j": "violet", "k": "silver", "l": "gold"}
# Mystery Blocksworld predicate names -> Blocksworld ones (configs/mystery_blocksworld.yaml)
MYSTERY_PREDICATES = {"craves": "on", "planet": "ontable", "province": "clear",
                      "harmony": "handempty", "pain": "holding"}


def round_robin(items_by_stratum, n, rng):
    """Shuffle each stratum, then take one item per stratum in turn until n are taken."""
    strata = sorted(items_by_stratum)
    pools = {s: rng.sample(items_by_stratum[s], len(items_by_stratum[s])) for s in strata}
    out = []
    while len(out) < n and any(pools.values()):
        for s in strata:
            if pools[s] and len(out) < n:
                out.append(pools[s].pop())
    return out


def natural_plan(task, stratum_of, gold_of, complexity_of):
    data = json.loads((NP_DIR / f"{task}.json").read_text(encoding="utf-8"))
    by_stratum = defaultdict(list)
    for key, item in data.items():
        by_stratum[stratum_of(item)].append(key)
    keys = round_robin(by_stratum, N_PER_TASK, random.Random(f"{SEED}|{task}"))
    rows = []
    for key in keys:
        item = data[key]
        rows.append({
            "id": key,
            "benchmark": task.split("_")[0],
            "system": NP_SYSTEM,
            "prompt": item["prompt_5shot"],
            "complexity": complexity_of(item),
            "gold": gold_of(item),
        })
    return rows


def parse_pddl(path):
    text = path.read_text(encoding="utf-8").lower()
    objects = re.search(r"\(:objects([^)]*)\)", text).group(1).split()
    init_part = text[text.index("(:init"):text.index("(:goal")]
    goal_part = text[text.index("(:goal"):]
    fact = re.compile(r"\(([a-z-]+)((?:\s+[a-z0-9]+)*)\s*\)")

    def facts(part):
        out = []
        for name, args in fact.findall(part):
            if name in (":init", ":goal", "and"):
                continue
            name = MYSTERY_PREDICATES.get(name, name)
            out.append([name] + args.split())
        return out

    return objects, facts(init_part), facts(goal_part)


def blocksworld(domain):
    prompts = json.loads((PB_DIR / "prompts" / domain / "task_1_plan_generation.json").read_text(encoding="utf-8"))
    inst_dir = PB_DIR / "instances" / "blocksworld" / ("mystery/generated_basic" if domain.startswith("mystery") else "generated_basic")
    rows = []
    for inst in prompts["instances"]:
        objects, init, goal = parse_pddl(inst_dir / f"instance-{inst['instance_id']}.pddl")
        gt = [l for l in inst["ground_truth_plan"].splitlines() if l.strip()]
        rows.append({
            "id": f"bw_{inst['instance_id']}",
            "benchmark": domain,
            "system": BW_SYSTEM,
            "prompt": inst["query"],
            "complexity": {"n_blocks": len(objects), "plan_length": len(gt)},
            "gold": {"objects": objects, "init": init, "goal": goal, "ground_truth_plan": gt},
        })
    return rows


def bw_sentence(facts, names):
    """Render facts the way PlanBench does: 'the red block is clear, ... and the hand is empty'."""
    order = {"clear": 0, "handempty": 1, "on": 2, "ontable": 3}
    text = []
    for f in sorted(facts, key=lambda f: order[f[0]]):
        if f[0] == "clear":
            text.append(f"the {names[f[1]]} block is clear")
        elif f[0] == "handempty":
            text.append("the hand is empty")
        elif f[0] == "on":
            text.append(f"the {names[f[1]]} block is on top of the {names[f[2]]} block")
        else:
            text.append(f"the {names[f[1]]} block is on the table")
    return text[0] if len(text) == 1 else ", ".join(text[:-1]) + " and " + text[-1]


def blocksworld_hard(one_shot_prefix):
    """The 110 larger instances (6-15 blocks) of Valmeekam et al. (2024), which ship without
    natural-language prompts. Rendered in the task 1 format with the same one-shot example."""
    root = HERE / "external" / "LLMs-Planning" / "llm_planning_analysis" / "instances" / "blocksworld_hard"
    info = json.loads((root / "info.json").read_text(encoding="utf-8"))
    names = dict(BW_COLORS, m="brown", n="pink", o="purple", p="teal")
    rows = []
    for i in range(1, 111):
        objects, init, goal = parse_pddl(root / "generated" / f"instance-{i}.pddl")
        statement = (f"[STATEMENT]\nAs initial conditions I have that, {bw_sentence(init, names)}.\n"
                     f"My goal is to have that {bw_sentence(goal, names)}.\n\nMy plan is as follows:\n\n[PLAN]")
        gt = [l for l in info[f"instance-{i}.pddl"]["plan"].splitlines() if l.strip()]
        rows.append({
            "id": f"bwhard_{i}",
            "benchmark": "blocksworld_hard",
            "system": BW_SYSTEM,
            "prompt": one_shot_prefix + statement,
            "complexity": {"n_blocks": len(objects), "plan_length": len(gt)},
            "gold": {"objects": objects, "init": init, "goal": goal, "ground_truth_plan": gt},
        })
    return rows


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} items")


def main():
    write("natural_plan_trip.jsonl", natural_plan(
        "trip_planning",
        stratum_of=lambda it: int(it["num_cities"]),
        gold_of=lambda it: {"cities": it["cities"], "durations": it["durations"], "golden_plan": it["golden_plan"]},
        complexity_of=lambda it: {"num_cities": int(it["num_cities"])},
    ))
    write("natural_plan_meeting.jsonl", natural_plan(
        "meeting_planning",
        stratum_of=lambda it: int(it["num_people"]),
        gold_of=lambda it: {"constraints": it["constraints"], "dist_matrix": it["dist_matrix"], "golden_plan": it["golden_plan"]},
        complexity_of=lambda it: {"num_people": int(it["num_people"])},
    ))
    write("natural_plan_calendar.jsonl", natural_plan(
        "calendar_scheduling",
        stratum_of=lambda it: (int(it["num_people"]), int(it["num_days"])),
        gold_of=lambda it: {"golden_plan": it["golden_plan"]},
        complexity_of=lambda it: {"num_people": int(it["num_people"]), "num_days": int(it["num_days"])},
    ))

    bw = blocksworld("blocksworld")
    mystery = {r["id"]: r for r in blocksworld("mystery_blocksworld")}
    order = random.Random(f"{SEED}|blocksworld").sample(bw, len(bw))
    write("blocksworld.jsonl", order)
    write("mystery_blocksworld.jsonl", [mystery[r["id"]] for r in order if r["id"] in mystery])

    # Domain description + the solved example of a standard task 1 prompt, up to its new [STATEMENT].
    prefix = bw[0]["prompt"][: bw[0]["prompt"].rfind("[STATEMENT]")]
    hard = blocksworld_hard(prefix)
    write("blocksworld_hard.jsonl", random.Random(f"{SEED}|blocksworld_hard").sample(hard, len(hard)))


if __name__ == "__main__":
    main()

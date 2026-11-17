"""
Score the answers saved by run_claude.py and report accuracy with 95% Wilson
confidence intervals, accuracy by problem size, a logistic regression of
correctness on problem size, and an exact McNemar test between Blocksworld and
Mystery Blocksworld on the instances answered in both.

Usage:
    python score.py                       # every results/*.jsonl
    python score.py --figure figures/accuracy_by_complexity.png

Scoring follows each benchmark's own rules:
  * Natural Plan: the parsing and exact-match logic of the official scripts in
    github.com/google-deepmind/natural-plan (Apache 2.0), copied below.
    Trip: cities and days must match the golden plan. Calendar: day, start and
    end must match. Meeting: the plan must be valid and meet as many friends as
    the golden plan.
  * Blocksworld: the plan is executed from the initial state; every action must
    be applicable and the goal must hold at the end (what VAL checks).
"""
import argparse
import datetime
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
FILES = {
    "trip": "natural_plan_trip.jsonl",
    "meeting": "natural_plan_meeting.jsonl",
    "calendar": "natural_plan_calendar.jsonl",
    "blocksworld": "blocksworld.jsonl",
    "mystery_blocksworld": "mystery_blocksworld.jsonl",
    "blocksworld_hard": "blocksworld_hard.jsonl",
}
COMPLEXITY = {"trip": "num_cities", "meeting": "num_people", "calendar": "num_people",
              "blocksworld": "plan_length", "mystery_blocksworld": "plan_length",
              "blocksworld_hard": "n_blocks"}
LABELS = {"trip": "NP Trip", "meeting": "NP Meeting", "calendar": "NP Calendar",
          "blocksworld": "Blocksworld", "mystery_blocksworld": "Mystery BW",
          "blocksworld_hard": "BW large"}


# ---------------------------------------------------------------- Natural Plan (official logic)

def trip_parse(response):
    pattern_visit = r"\d+-\d+"
    pattern_flight = r".*Day (\d+).*from (\w+) to (\w+)"
    pattern_days = r"European cities for (\d+) days"
    days, flights, flight_days = [], [], []
    total_days = None
    for piece in response.split("\n"):
        days_match = re.findall(pattern_days, piece)
        if days_match:
            total_days = int(days_match[0])
        visit_match = re.findall(pattern_visit, piece)
        if visit_match:
            days.append(visit_match[0])
            end_day = int(visit_match[0].split("-")[1])
            if end_day == total_days:  # end of the plan, do not parse alternative plans
                break
        flight_match = re.findall(pattern_flight, piece)
        if flight_match:
            flights.append(flight_match[0])
    visit_cities, parsed_plan = [], []
    for flight_day, begin_city, end_city in flights:
        flight_days.append(int(flight_day))
        if not visit_cities:
            visit_cities += [begin_city, end_city]
        else:
            visit_cities.append(end_city)
    if not days or not flights or not visit_cities:
        return []
    last_day = int(days[-1].split("-")[1])
    flight_days = [1] + flight_days + [last_day]
    for i, visit_city in enumerate(visit_cities):
        parsed_plan.append((visit_city, flight_days[i + 1] - flight_days[i] + 1))
    return parsed_plan


def score_trip(text, gold):
    plan = trip_parse(text)
    stays = [x for x in gold["cities"].split("**") if x]
    days = [int(x) for x in gold["durations"].split("**") if x]
    num_match = 0
    for i in range(min(len(stays), len(plan))):
        if stays[i] == plan[i][0] and days[i] == plan[i][1]:
            num_match += 1
        else:
            break
    return num_match == len(stays), bool(plan)


def _hour(hr):
    return float(hr.split(":")[0]) + (0.5 if hr.split(":")[1] == "30" else 0.0)


def calendar_parse(response):
    found = re.findall(r"[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+", response)
    if not found:
        return "", -1, -1
    day, hours = found[0].split(",")[0].strip(), found[0].split(",")[1].strip()
    return day, _hour(hours.split("-")[0].strip()), _hour(hours.split("-")[1].strip())


def score_calendar(text, gold):
    pred = calendar_parse(text)
    return pred == calendar_parse(gold["golden_plan"]), pred[0] != ""


def _time(s):
    return datetime.datetime.strptime(s, "%I:%M%p")


def meeting_validate(plan, gold):
    start_location, initial_time = gold["constraints"][0]
    people = {}
    for name, location, times, minutes in gold["constraints"][1:]:
        people[name] = {"location": location, "start": _time(times.split("to")[0].strip()),
                        "end": _time(times.split("to")[1].strip()), "minutes": minutes}
    dist = gold["dist_matrix"]
    met, score = set(), 0
    loc, now = start_location, _time(initial_time)
    for step in plan:
        try:
            if step.startswith("You start"):
                continue
            elif step.startswith("You travel"):
                dest = step.split("travel to ")[1].split(" in")[0].strip()
                now += datetime.timedelta(minutes=dist[loc][dest])
                loc = dest
            elif step.startswith("You wait"):
                until = _time(step.split("wait until ")[1].split(".")[0].strip())
                if until <= now:
                    raise ValueError("Cannot go backwards in time")
                now = until
            elif step.startswith("You meet"):
                person = step.split("meet ")[1].split(" for")[0].strip()
                if person in met:
                    raise ValueError("already met")
                met.add(person)
                end = now + datetime.timedelta(minutes=people[person]["minutes"])
                if loc == people[person]["location"] and now >= people[person]["start"] and end <= people[person]["end"]:
                    score += 1
                    now = end
                else:
                    raise ValueError("Invalid meeting time or location")
            else:
                raise ValueError("Unknown plan format")
        except (ValueError, KeyError, IndexError):
            break
    return score


def meeting_parse(text):
    if "SOLUTION:" in text:
        text = text[text.find("SOLUTION:") + len("SOLUTION:"):].strip()
    return [s.strip() for s in text.split(".") if s.strip()]


def score_meeting(text, gold):
    plan = meeting_parse(text)
    return meeting_validate(plan, gold) == meeting_validate(gold["golden_plan"], gold), bool(plan)


# ---------------------------------------------------------------- Blocksworld

COLOR_TO_LETTER = {"red": "a", "blue": "b", "orange": "c", "yellow": "d", "white": "e", "magenta": "f",
                   "black": "g", "cyan": "h", "green": "i", "violet": "j", "silver": "k", "gold": "l",
                   "brown": "m", "pink": "n", "purple": "o", "teal": "p"}
BW_ACTION = re.compile(
    r"\b(pick up|put down|unstack|stack)\s+(?:the\s+)?(\w+)\s+block"
    r"(?:\s+(?:from\s+)?on\s+top\s+of\s+(?:the\s+)?(\w+)\s+block)?", re.I)
MYSTERY_ACTION = re.compile(
    r"\b(attack|succumb|feast|overcome)\s+object\s+([a-l])\b(?:\s+from\s+(?:another\s+)?object\s+([a-l])\b)?", re.I)
ACTION_NAMES = {"pick up": "pick-up", "put down": "put-down", "unstack": "unstack", "stack": "stack",
                "attack": "pick-up", "succumb": "put-down", "feast": "unstack", "overcome": "stack"}


def bw_parse(text, mystery):
    """Return the plan as [(action, x, y)], taken from the last [PLAN] ... [PLAN END] block,
    or from the last run of lines that contain actions if the tags are missing."""
    pattern = MYSTERY_ACTION if mystery else BW_ACTION
    segment = text
    if "[PLAN END]" in segment:
        segment = segment[: segment.rfind("[PLAN END]")]
    if "[PLAN]" in segment:
        segment = segment[segment.rfind("[PLAN]") + len("[PLAN]"):]
    else:
        lines, run = segment.splitlines(), []
        for line in reversed(lines):
            if pattern.search(line):
                run.append(line)
            elif line.strip() and run:
                break
        segment = "\n".join(reversed(run))
    plan = []
    for name, x, y in pattern.findall(segment):
        name = ACTION_NAMES[name.lower()]
        if not mystery:
            x, y = COLOR_TO_LETTER.get(x.lower(), x.lower()), COLOR_TO_LETTER.get((y or "").lower(), (y or "").lower())
        plan.append((name, x.lower(), (y or "").lower()))
    return plan


def bw_execute(plan, gold):
    state = {tuple(f) for f in gold["init"]}
    for name, x, y in plan:
        if name == "pick-up":
            pre, add, dele = [("clear", x), ("ontable", x), ("handempty",)], [("holding", x)], [("clear", x), ("ontable", x), ("handempty",)]
        elif name == "put-down":
            pre, add, dele = [("holding", x)], [("clear", x), ("handempty",), ("ontable", x)], [("holding", x)]
        elif name == "stack":
            pre, add, dele = [("clear", y), ("holding", x)], [("handempty",), ("clear", x), ("on", x, y)], [("clear", y), ("holding", x)]
        else:  # unstack
            pre, add, dele = [("on", x, y), ("clear", x), ("handempty",)], [("holding", x), ("clear", y)], [("on", x, y), ("clear", x), ("handempty",)]
        if not all(p in state for p in pre):
            return False
        state = (state - set(dele)) | set(add)
    return all(tuple(g) in state for g in gold["goal"])


def score_bw(text, gold, mystery):
    plan = bw_parse(text, mystery)
    return bool(plan) and bw_execute(plan, gold), bool(plan)


SCORERS = {
    "trip": score_trip, "meeting": score_meeting, "calendar": score_calendar,
    "blocksworld": lambda t, g: score_bw(t, g, False),
    "mystery_blocksworld": lambda t, g: score_bw(t, g, True),
    "blocksworld_hard": lambda t, g: score_bw(t, g, False),
}


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def binom_two_sided(k, n):
    """Exact two-sided binomial test of k successes in n trials against p = 0.5."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] * (1 + 1e-9)))


def binom_one_sided_below(k, n, p0):
    """P(X <= k) for X ~ Bin(n, p0): small values mean accuracy is significantly below p0."""
    return sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k + 1))


def logistic(xs, ys, iters=50):
    """Logistic regression y ~ b0 + b1*x by Newton-Raphson. Returns b1, its SE, Wald z and p."""
    b0 = b1 = 0.0
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-(b0 + b1 * x)))
            g0 += y - p
            g1 += (y - p) * x
            w = p * (1 - p)
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            return None
        b0 += (h11 * g0 - h01 * g1) / det
        b1 += (h00 * g1 - h01 * g0) / det
        if abs(b1) > 50:  # perfect separation (e.g. everything correct)
            return None
    se = math.sqrt(h00 / det)
    z = b1 / se
    return b1, se, z, math.erfc(abs(z) / math.sqrt(2))


# ---------------------------------------------------------------- main

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def score_all(results_dir):
    scored = {}
    for bench, fname in FILES.items():
        path = Path(results_dir) / fname
        if not path.exists():
            continue
        questions = {q["id"]: q for q in load_jsonl(HERE / "subsets" / fname)}
        rows = []
        for r in load_jsonl(path):
            q = questions[r["id"]]
            correct, parsed = SCORERS[bench](r["text"], q["gold"])
            rows.append({"id": r["id"], "correct": bool(correct), "parsed": parsed,
                         "x": q["complexity"][COMPLEXITY[bench]], "output_tokens": r["usage"].get("output_tokens"),
                         "duration_s": r.get("duration_s")})
        scored[bench] = rows
    return scored


def report(scored):
    out = [f"Updated {datetime.datetime.now():%Y-%m-%d %H:%M}\n"]
    out.append("| Benchmark | n | Correct | Accuracy | 95% CI | p (acc < 90%) | Unparsed |")
    out.append("|---|---|---|---|---|---|---|")
    for bench, rows in scored.items():
        n, k = len(rows), sum(r["correct"] for r in rows)
        lo, hi = wilson(k, n)
        out.append(f"| {LABELS[bench]} | {n} | {k} | {100 * k / n:.1f}% | {100 * lo:.1f}–{100 * hi:.1f}% "
                   f"| {binom_one_sided_below(k, n, 0.9):.3g} | {sum(not r['parsed'] for r in rows)} |")
    np_benches = [b for b in ("trip", "meeting", "calendar") if b in scored]
    if np_benches:
        accs = [sum(r["correct"] for r in scored[b]) / len(scored[b]) for b in np_benches]
        out.append(f"\nNatural Plan mean of task accuracies ({', '.join(np_benches)}): {100 * sum(accs) / len(accs):.1f}%")

    out.append("\nAccuracy by problem size")
    for bench, rows in scored.items():
        groups = defaultdict(list)
        for r in rows:
            groups[r["x"]].append(r["correct"])
        cells = ", ".join(f"{x}: {sum(v)}/{len(v)}" for x, v in sorted(groups.items()))
        fit = logistic([r["x"] for r in rows], [int(r["correct"]) for r in rows])
        reg = (f"logistic b = {fit[0]:.2f} (SE {fit[1]:.2f}), OR = {math.exp(fit[0]):.2f}, z = {fit[2]:.2f}, p = {fit[3]:.3g}"
               if fit else "logistic regression not estimable (all correct, or errors only at the largest sizes)")
        out.append(f"- {LABELS[bench]} by {COMPLEXITY[bench]}: {cells}; {reg}")

    if "blocksworld" in scored and "mystery_blocksworld" in scored:
        a = {r["id"]: r["correct"] for r in scored["blocksworld"]}
        b = {r["id"]: r["correct"] for r in scored["mystery_blocksworld"]}
        shared = sorted(set(a) & set(b))
        if shared:
            only_a = sum(a[i] and not b[i] for i in shared)
            only_b = sum(b[i] and not a[i] for i in shared)
            acc_a = sum(a[i] for i in shared) / len(shared)
            acc_b = sum(b[i] for i in shared) / len(shared)
            out.append(f"\nPaired comparison on {len(shared)} shared instances: Blocksworld {100 * acc_a:.1f}% vs "
                       f"Mystery {100 * acc_b:.1f}%; discordant pairs {only_a} (only BW correct) / {only_b} "
                       f"(only Mystery correct); exact McNemar p = {binom_two_sided(only_a, only_a + only_b):.3g}")

    out.append("\nMean output tokens / seconds per item")
    for bench, rows in scored.items():
        toks = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
        secs = [r["duration_s"] for r in rows if r["duration_s"]]
        out.append(f"- {LABELS[bench]}: {sum(toks) / max(1, len(toks)):.0f} tokens, {sum(secs) / max(1, len(secs)):.0f} s")
    return "\n".join(out)


def figure(scored, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benches = list(scored)
    fig, axes = plt.subplots(1, len(benches), figsize=(3.2 * len(benches), 3.2), sharey=True, squeeze=False)
    for ax, bench in zip(axes[0], benches):
        groups = defaultdict(list)
        for r in scored[bench]:
            groups[r["x"]].append(r["correct"])
        xs = sorted(groups)
        acc = [sum(groups[x]) / len(groups[x]) for x in xs]
        ci = [wilson(sum(groups[x]), len(groups[x])) for x in xs]
        ax.errorbar(xs, [100 * a for a in acc], yerr=[[max(0.0, 100 * (a - c[0])) for a, c in zip(acc, ci)],
                                                      [max(0.0, 100 * (c[1] - a)) for a, c in zip(acc, ci)]],
                    fmt="o-", capsize=3, color="#2f6db3")
        ax.axhline(90, ls="--", color="#c0392b", lw=1)
        n, k = len(scored[bench]), sum(r["correct"] for r in scored[bench])
        ax.set_title(f"{LABELS[bench]}\n{k}/{n} = {100 * k / n:.0f}%", fontsize=10)
        ax.set_xlabel(COMPLEXITY[bench].replace("_", " "))
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy (%)")
    axes[0][0].text(0.02, 0.80, "AGI threshold 90%", transform=axes[0][0].transAxes, color="#c0392b", fontsize=8,
                    va="bottom")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"figure saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--figure", default="")
    ap.add_argument("--summary", default=str(HERE / "results" / "summary.md"))
    args = ap.parse_args()
    scored = score_all(args.results)
    if not scored:
        raise SystemExit(f"no results in {args.results}; run run_claude.py first")
    text = report(scored)
    print(text)
    Path(args.summary).write_text(text + "\n", encoding="utf-8")
    if args.figure:
        figure(scored, args.figure)


if __name__ == "__main__":
    main()

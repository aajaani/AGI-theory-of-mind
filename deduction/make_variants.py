"""
Build memorization and robustness variants of the same LogiQA 2.0 items your baseline uses.

It selects items exactly like run_logiqa_claude.py (shuffle with --seed, take first --n),
then writes one JSONL file per variant into --outdir. Every variant keeps the original
"id" and "type" so results can be compared item by item and per category.

Usage:
    python make_variants.py --data LogiQA2.0\\logiqa\\DATA\\LOGIQA\\test.txt --n 100 --seed 42

Multiple-choice variants (run each with run_logiqa_claude.py):
    baseline.jsonl          the 100 original items, unchanged
    shuffle_options.jsonl   options in a new random order, answer re-mapped        [memorization]
    answer_moved.jsonl      options rotated so the correct one ALWAYS changes slot [memorization]
    no_passage.jsonl        passage removed; question + options only               [memorization / shortcuts]
    renamed_entities.jsonl  person names replaced by invented names                [robustness]
    distractors.jsonl       two irrelevant sentences added to the passage          [robustness]
    typos.jsonl             small letter swaps in the passage (logic words kept)   [robustness]
    combined.jsonl          renamed + distractors + shuffled options together      [robustness]
    replace_answer.jsonl    correct option replaced by "None of the other options
                            is correct" (same slot)                                [memorization]
    shuffle_replace.jsonl   replace_answer + options shuffled                      [memorization]
    (the last two follow LogiQA-plus, Bao et al. 2024, arXiv:2310.09430)

Memorization probes (run with run_probes.py):
    probes.jsonl            passage first half -> second half; passage+question -> options
"""
import argparse
import json
import random
import re
from pathlib import Path

LETTERS = "ABCD"

# ---------------------------------------------------------------- helpers

def load_subset(path, n, seed):
    data = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    random.Random(seed).shuffle(data)  # identical to run_logiqa_claude.py
    return data[:n]


def copy_item(ex, variant):
    out = {k: ex[k] for k in ("id", "answer", "text", "question", "options") if k in ex}
    out["type"] = ex.get("type", {})
    out["variant"] = variant
    return out


def apply_perm(item, perm):
    """perm[new_position] = original_position."""
    item["options"] = [item["options"][p] for p in perm]
    item["answer"] = perm.index(item["answer"])
    item["perm"] = perm
    return item


def strip_label(opt):
    # some options carry their own "A." / "B." prefix; drop it so shuffled labels don't clash
    return re.sub(r"^\s*[A-Da-d]\s*[\.\)、:：]\s*", "", opt)


# ---------------------------------------------------------------- variants

def v_shuffle(ex, rng):
    item = copy_item(ex, "shuffle_options")
    item["options"] = [strip_label(o) for o in item["options"]]
    perm = list(range(len(item["options"])))
    while perm == list(range(len(perm))):
        rng.shuffle(perm)
    return apply_perm(item, perm)


def v_answer_moved(ex, rng):
    item = copy_item(ex, "answer_moved")
    item["options"] = [strip_label(o) for o in item["options"]]
    n = len(item["options"])
    k = rng.randint(1, n - 1)  # rotation by k != 0 guarantees the gold option moves
    perm = [(i + k) % n for i in range(n)]
    return apply_perm(item, perm)


def v_no_passage(ex, rng):
    item = copy_item(ex, "no_passage")
    item["text"] = ("[The passage for this question has been removed. "
                    "Choose the most likely answer anyway.]")
    return item


# --- renaming people ------------------------------------------------------

CHINESE_SURNAMES = """Zhang Wang Li Liu Chen Yang Zhao Huang Zhou Wu Xu Sun Hu Zhu Gao Lin Guo Ma Luo
Liang Zheng Xie   Feng Yu Dong Cheng Cao Yuan Deng Fu Shen Zeng Peng Su Lu Jiang Cai Jia
Ding Wei Xue Yan Pan Du Dai Xia Zhong Tian Ren Yao Tan Liao Zou Xiong  Hao Cui Kang Mao
Qiu  Gu Hou Shao Meng Duan Qian Yin Qiao Lai Gong Wen San Si Mei    Jun""".split()
WESTERN_NAMES = """John Mary Tom Jack Peter Paul David James Robert Michael William Richard Thomas
Charles George Henry Alice Anna Emma Lucy Linda Susan Sarah Jane Kate Helen Laura Julia Lisa Smith
Johnson Jones Miller Davis Wilson Taylor Anderson Lee Clark Lewis Walker Harris Martin""".split()
# Left out on purpose (also common words / places / dynasties): He, Long, Han, Tang, Qin, Jin,
# Hong, Kong, Ming, Hua, Song, White, Brown.
PERSON_WORDS = set(CHINESE_SURNAMES) | set(WESTERN_NAMES)
HONORIFIC = r"(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Lao|Xiao|Old|Little|Uncle|Aunt|Teacher)\.?"

NONCE_NAMES = """Davor Kelmi Orsa Talvin Brenna Qessa Morvik Ilka Fendro Sabeth Ulmar Renka Tobrin
Velsa Garrow Nilda Pashko Yorvel Emba Krisol Halvard Zenna Dorvan Lusk Marit Oskel Pella Ravik
Sindra Tuomo Valka Wenlo Abri Corvin Eskel Fiora Grell Hanno Isve Jorren""".split()


def find_person_names(texts):
    joined = "\n".join(texts)
    names = set()
    for w in set(re.findall(r"\b[A-Z][a-z]+\b", joined)):
        if w in PERSON_WORDS:
            names.add(w)
    for w in re.findall(HONORIFIC + r"\s+([A-Z][a-z]+)", joined):
        if w not in {"The", "A", "An"}:
            names.add(w)
    return sorted(names)


def v_renamed(ex, rng, variant="renamed_entities"):
    item = copy_item(ex, variant)
    fields = [item["text"], item["question"], *item["options"]]
    names = find_person_names(fields)
    pool = NONCE_NAMES[:]
    rng.shuffle(pool)
    mapping = {n: pool[i % len(pool)] for i, n in enumerate(names)}

    def sub(s):
        for old, new in mapping.items():
            s = re.sub(rf"\b{re.escape(old)}\b", new, s)
        return s

    item["text"] = sub(item["text"])
    item["question"] = sub(item["question"])
    item["options"] = [sub(o) for o in item["options"]]
    item["renames"] = mapping
    return item


# --- distractors -------------------------------------------------------------

DISTRACTORS = [
    "The document describing this situation was printed on recycled paper.",
    "Light rain was reported in the region that afternoon.",
    "A nearby cafe had recently changed its opening hours.",
    "The original notes were kept in a grey folder on a shelf.",
    "The meeting room mentioned in the records was repainted last spring.",
    "A local radio station was playing classical music at the time.",
    "The building's front door had been replaced the previous year.",
    "Several people in the area had recently taken up gardening.",
]


def v_distractors(ex, rng, variant="distractors"):
    item = copy_item(ex, variant) if isinstance(ex, dict) and "variant" not in ex else ex
    item["variant"] = variant
    first, last = rng.sample(DISTRACTORS, 2)
    item["text"] = f"{first} {item['text'].strip()} {last}"
    return item


# --- typos ------------------------------------------------------------------

PROTECTED = set("""not no none nor never neither either only unless if then all every each any some
most least more less than and or but because therefore must cannot can might may should would
true false same different both without except""".split())


def typo_word(w, rng):
    core = re.match(r"^([A-Za-z]+)(.*)$", w)
    if not core:
        return w
    letters, tail = core.groups()
    if len(letters) < 5 or letters.lower() in PROTECTED or letters[0].isupper():
        return w
    i = rng.randint(1, len(letters) - 3)  # swap two inner letters, keep first/last
    letters = letters[:i] + letters[i + 1] + letters[i] + letters[i + 2:]
    return letters + tail


def v_typos(ex, rng, rate=0.06):
    item = copy_item(ex, "typos")
    words = item["text"].split(" ")
    item["text"] = " ".join(typo_word(w, rng) if rng.random() < rate else w for w in words)
    return item


def v_combined(ex, rng):
    item = v_renamed(ex, rng, variant="combined")
    item = v_distractors(item, rng, variant="combined")
    item["options"] = [strip_label(o) for o in item["options"]]
    perm = list(range(len(item["options"])))
    while perm == list(range(len(perm))):
        rng.shuffle(perm)
    return apply_perm(item, perm)


# --- replace the correct answer ----------------------------------------------

NONE_OPTION = "None of the other options is correct."

# Questions asking for the "most"/"best" option are relative: once the real answer is removed,
# one of the remaining options may arguably become "the best", so the new label is debatable.
RELATIVE_Q = re.compile(r"\b(most|best|strongest|weakest|least|mainly|primarily|"
                        r"most likely|most appropriate|most accurate)\b", re.IGNORECASE)


def v_replace_answer(ex, rng, variant="replace_answer"):
    item = copy_item(ex, variant)
    item["options"] = [strip_label(o) for o in item["options"]]
    item["options"][item["answer"]] = NONE_OPTION  # answer index stays the same
    item["relative_q"] = bool(RELATIVE_Q.search(item["question"]))
    return item


def v_shuffle_replace(ex, rng):
    item = v_replace_answer(ex, rng, variant="shuffle_replace")
    perm = list(range(len(item["options"])))
    while perm == list(range(len(perm))):
        rng.shuffle(perm)
    return apply_perm(item, perm)


# ---------------------------------------------------------------- probes

def make_probes(ex):
    words = ex["text"].split()
    half = max(5, len(words) // 2)
    return [
        {"id": ex["id"], "probe": "completion",
         "prefix": " ".join(words[:half]), "target": " ".join(words[half:half + 60]),
         "type": ex.get("type", {})},
        {"id": ex["id"], "probe": "options",
         "text": ex["text"], "question": ex["question"], "target_options": ex["options"],
         "type": ex.get("type", {})},
    ]


# ---------------------------------------------------------------- main

VARIANTS = {
    "shuffle_options": v_shuffle,
    "answer_moved": v_answer_moved,
    "no_passage": v_no_passage,
    "renamed_entities": v_renamed,
    "distractors": v_distractors,
    "typos": v_typos,
    "combined": v_combined,
    "replace_answer": v_replace_answer,
    "shuffle_replace": v_shuffle_replace,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="LogiQA2.0/logiqa/DATA/LOGIQA/test.txt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42, help="must match the baseline run")
    ap.add_argument("--outdir", default="variants")
    args = ap.parse_args()

    subset = load_subset(args.data, args.n, args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, items):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  {name + '.jsonl':26s} {len(items)} items")

    print(f"Selected {len(subset)} items (seed {args.seed}). Writing to {out}/")
    write("baseline", [copy_item(ex, "baseline") for ex in subset])
    for name, fn in VARIANTS.items():
        rng = random.Random(f"{args.seed}-{name}")  # each variant reproducible on its own
        items = [fn(ex, rng) for ex in subset]
        write(name, items)
        if name == "renamed_entities":
            changed = sum(bool(it["renames"]) for it in items)
            print(f"    (names found and replaced in {changed}/{len(items)} items; "
                  "others are unchanged)")
        if name == "replace_answer":
            rel = sum(it["relative_q"] for it in items)
            print(f"    ({rel}/{len(items)} are 'most/best'-type questions where the new label "
                  "is debatable; reported separately by compare_runs.py)")
    write("probes", [p for ex in subset for p in make_probes(ex)])


if __name__ == "__main__":
    main()
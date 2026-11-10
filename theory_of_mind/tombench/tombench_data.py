import json
import math
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The 8 tasks of the paper's task view (Table 2); human average 85.4% is the
# unweighted mean over these. The other 12 files are ability-only extras.
TASK_FILES = [
    "Unexpected Outcome Test",
    "Scalar Implicature Test",
    "Persuasion Story Task",
    "False Belief Task",
    "Ambiguous Story Task",
    "Hinting Task Test",
    "Strange Story Task",
    "Faux-pas Recognition Test",
]

HUMAN_TASK_ACC = {
    "Unexpected Outcome Test": 89.3,
    "Scalar Implicature Test": 75.5,
    "Persuasion Story Task": 70.0,
    "False Belief Task": 86.8,
    "Ambiguous Story Task": 95.0,
    "Hinting Task Test": 97.1,
    "Strange Story Task": 89.2,
    "Faux-pas Recognition Test": 80.4,
}
HUMAN_TASK_AVG = 85.4

LETTERS = "ABCD"
OPTION_PREFIX = re.compile(r"^\s*[A-D]\s*[\.．、]\s*")


def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _options(d, keys):
    opts = [OPTION_PREFIX.sub("", str(d[k])).strip() for k in keys if not _missing(d.get(k))]
    if len(opts) not in (2, 4):
        raise ValueError(f"unexpected option count {len(opts)}")
    return opts


def normalize_ability(raw):
    a = raw.strip().replace("Non-literal communication", "Non-Literal Communication")
    # Questions labelled "<false belief> Belief: Second-order beliefs" are the
    # paper's Second-order beliefs ability (Table 18: 200 questions).
    if a.endswith("Belief: Second-order beliefs"):
        return "Belief: Second-order beliefs"
    if a.startswith("Desire: Desires influence on"):
        return "Desire: Desires influence on actions/emotions"
    return a


def _apply_fixes(file, row, d, rec):
    fixes = rec["fixes"]

    raw_answer = d["答案\nANSWER"]
    answer = raw_answer.strip().rstrip(".").strip()
    if answer != raw_answer:
        fixes.append(f"answer key {raw_answer!r} -> {answer!r}")
    rec["answer"] = answer

    # English question/options copied from the previous row (the Li Hua story).
    if file == "Strange Story Task" and row == 292:
        assert d["问题"] == "赵阳说的是真的吗？", "Strange Story row 292 changed upstream"
        rec["en"]["question"] = "Is what Zhao Yang says true?"
        rec["en"]["options"] = ["Yes", "No"]
        fixes.append("English question/options replaced (copied from row 291 by mistake)")

    # 生气 and 愤怒 were both translated as "Angry".
    en, zh = rec["en"]["options"], rec["zh"]["options"]
    if len(set(en)) < len(en):
        for i, z in enumerate(zh):
            if z == "愤怒" and en[i] == "Angry":
                en[i] = "Furious"
                fixes.append("duplicate English option 'Angry' (愤怒) -> 'Furious'")
        if len(set(en)) < len(en):
            raise ValueError(f"{file}#{row}: unresolved duplicate English options {en}")

    if len(rec["en"]["options"]) != len(rec["zh"]["options"]):
        raise ValueError(f"{file}#{row}: zh/en option counts differ")
    if LETTERS.index(rec["answer"]) >= len(rec["en"]["options"]):
        raise ValueError(f"{file}#{row}: answer {rec['answer']} out of range")


def load_file(file):
    records = []
    story = -1
    with open(os.path.join(DATA_DIR, f"{file}.jsonl"), encoding="utf-8") as f:
        for row, line in enumerate(f):
            d = json.loads(line)
            if d["序号\nINDEX"] == 1 or story < 0:
                story += 1
            raw_ability = d["能力\nABILITY"]
            ability = normalize_ability(raw_ability)
            rec = {
                "qid": f"{file}#{row}",
                "file": file,
                "row": row,
                "is_task": file in TASK_FILES,
                "index": d["序号\nINDEX"],
                "story_id": f"{file}#s{story}",
                "ability": ability,
                "ability_raw": raw_ability,
                "category": ability.split(":")[0],
                "en": {
                    "story": d["STORY"],
                    "question": d["QUESTION"],
                    "options": _options(d, ["OPTION-A", "OPTION-B", "OPTION-C", "OPTION-D"]),
                },
                "zh": {
                    "story": d["故事"],
                    "question": d["问题"],
                    "options": _options(d, ["选项A", "选项B", "选项C", "选项D"]),
                },
                "fixes": [],
            }
            _apply_fixes(file, row, d, rec)
            records.append(rec)
    return records


def all_files():
    return sorted(f[:-6] for f in os.listdir(DATA_DIR) if f.endswith(".jsonl"))


def load_all():
    records = []
    for file in all_files():
        records.extend(load_file(file))
    return records


def read_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_questions(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import collections

    recs = load_all()
    print(f"{len(recs)} questions, {len({r['story_id'] for r in recs})} stories")
    print(f"{sum(r['is_task'] for r in recs)} in the 8 task files")
    print(f"{len({r['ability'] for r in recs})} abilities, {len({r['category'] for r in recs})} categories")
    for r in recs:
        for fx in r["fixes"]:
            print(f"fix {r['qid']}: {fx}")
    for a, n in sorted(collections.Counter(r["ability"] for r in recs).items()):
        print(f"  {n:4d}  {a}")

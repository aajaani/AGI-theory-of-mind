"""Disguise an ARC task: random colour mapping + random flip, applied to every grid.

Usage:
    python3 disguise_task.py input.json output.json [--seed 42]
"""
import argparse
import json
import random
from pathlib import Path


def flip(grid, kind):
    if kind == "mirror_horizontal":   # left <-> right
        return [row[::-1] for row in grid]
    if kind == "mirror_vertical":     # top <-> bottom
        return grid[::-1]
    if kind == "rotate_180":
        return [row[::-1] for row in grid[::-1]]


def recolour(grid, mapping):
    return [[mapping[cell] for cell in row] for row in grid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = json.loads(Path(args.input).read_text())

    # Same seed + same file name always gives the same disguise.
    rng = random.Random(f"{args.seed}-{Path(args.input).name}")

    # All colours 0-9 are shuffled, including 0.
    shuffled = list(range(10))
    rng.shuffle(shuffled)
    mapping = dict(zip(range(10), shuffled))

    kind = rng.choice(["mirror_horizontal", "mirror_vertical", "rotate_180"])

    def transform(grid):
        return flip(recolour(grid, mapping), kind)

    # Apply to every grid, including test outputs, so the answer stays correct.
    for split in ("train", "test"):
        for pair in task[split]:
            for key in ("input", "output"):
                if key in pair:
                    pair[key] = transform(pair[key])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(task))
    print(f"{Path(args.input).name}: {kind}, colours {mapping}")


if __name__ == "__main__":
    main()

"""
summarize.py — Aggregate finished runs into the numbers reported in the paper.

For every run, the round with the highest client-averaged accuracy is taken;
the table reports mean ± std of that accuracy over the available seeds.

Usage:
    python summarize.py                       # reads ./output
    python summarize.py --output_dir ./output --alpha 0.5
"""

import argparse
import glob
import json
import os

import numpy as np

TASKS = ["mnli", "qnli", "sst2", "qqp"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    means = []
    for task in TASKS:
        paths = sorted(glob.glob(os.path.join(
            args.output_dir, "glue", task, f"alpha{args.alpha}", "seed*", "results.json")))
        if not paths:
            continue
        best = []
        for p in paths:
            with open(p) as f:
                scores = json.load(f)
            avgs = np.mean(list(scores.values()), axis=0)     # per-round average accuracy
            best.append(avgs.max())
        means.append(np.mean(best))
        print(f"{task.upper():6s} {100 * np.mean(best):.2f} ± {100 * np.std(best):.2f}"
              f"   ({len(paths)} seeds)")
    if len(means) == len(TASKS):
        print(f"{'Avg':6s} {100 * np.mean(means):.2f}")


if __name__ == "__main__":
    main()

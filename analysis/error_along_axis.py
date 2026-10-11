"""Where along the body does an organ go wrong?

    python analysis/error_along_axis.py data/SEGTHOR_zscore/val/gt \
        results/zscore_unet_k16_aug_seed* --organ esophagus --figure esophagus_axis.png

For every patient the organ's extent in the label (first to last slice that
contains it) is cut into equal parts along the scan axis. In each part the
Dice, the missed volume and the falsely added volume are pooled over the
slices, then averaged over patients and runs. The fraction of slices that
also contain trachea marks which end of the organ is the upper one.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ORGANS = ["esophagus", "heart", "trachea", "aorta"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--organ", default="esophagus", choices=ORGANS)
    parser.add_argument("--parts", type=int, default=5)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    k = ORGANS.index(args.organ) + 1
    trachea = ORGANS.index("trachea") + 1
    by_patient = defaultdict(list)
    for p in sorted(args.gt_dir.glob("*.png")):
        by_patient[p.stem.rsplit("_", 1)[0]].append(p.name)

    # [run, patient, part] -> tp, fp, fn, slices with trachea, slices
    stats = np.zeros((len(args.runs), len(by_patient), args.parts, 5))
    for i, run in enumerate(args.runs):
        for j, names in enumerate(by_patient.values()):
            gts = [np.array(Image.open(args.gt_dir / n)) // 63 for n in names]
            has = [bool((g == k).any()) for g in gts]
            if not any(has):
                continue
            first, last = has.index(True), len(has) - 1 - has[::-1].index(True)
            for z in range(first, last + 1):
                part = min(args.parts - 1, (z - first) * args.parts // (last - first + 1))
                pred = np.array(Image.open(run / "best_epoch" / "val" / names[z])) // 63
                g = gts[z]
                stats[i, j, part] += [((pred == k) & (g == k)).sum(), ((pred == k) & (g != k)).sum(),
                                      ((pred != k) & (g == k)).sum(), (g == trachea).any(), 1]

    tp, fp, fn, with_trachea, slices = np.moveaxis(stats, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * tp / (2 * tp + fp + fn)
        missed = fn / (tp + fn)
        added = fp / (tp + fn)
    mean = lambda a: np.nanmean(a, axis=(0, 1))
    upper = mean(with_trachea / slices)

    print(f"{args.organ}: label extent cut into {args.parts} equal parts, "
          f"{len(by_patient)} patients x {len(args.runs)} runs\n")
    print(f"{'part':>5s} {'Dice':>7s} {'missed':>8s} {'added':>7s} {'slices with trachea':>20s}")
    for part in range(args.parts):
        print(f"{part + 1:5d} {mean(dice)[part]:7.3f} {mean(missed)[part]:8.1%} "
              f"{mean(added)[part]:7.1%} {upper[part]:19.0%}")

    if args.figure:
        import matplotlib.pyplot as plt
        x = np.arange(1, args.parts + 1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, mean(dice), "o-", label="Dice")
        ax.plot(x, mean(missed), "s--", label="missed (share of true volume)")
        ax.plot(x, mean(added), "^--", label="falsely added (share of true volume)")
        ax.set_xticks(x)
        ax.set_xlabel(f"part of the {args.organ} along the scan (trachea present: "
                      + ", ".join(f"{u:.0%}" for u in upper) + ")")
        ax.set_ylim(0, 1)
        ax.legend()
        worst = int(np.nanargmin(mean(dice))) + 1
        ax.set_title(f"The {args.organ} is segmented worst in part {worst} of its length")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"\nSaved {args.figure}")


if __name__ == "__main__":
    main()

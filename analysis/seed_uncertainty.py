"""Does disagreement between seeds point at the model's errors?

    python analysis/seed_uncertainty.py data/SEGTHOR_zscore/val/gt \
        results/zscore_unet_k16_aug_seed1 results/zscore_unet_k16_aug_seed2 \
        results/zscore_unet_k16_aug_seed3 --img-dir data/SEGTHOR_zscore/val/img \
        --figure uncertainty.png

Three runs of the same configuration that differ only in their seed are three
equally good models. A voxel where they disagree is one the configuration is
unsure about. For every organ this reports how the errors of each seed split
between voxels where all seeds agree and voxels where they do not. If most
errors sit in the small disagreeing region, disagreement is a usable map of
where a radiologist should look.

Only voxels that any seed or the label assigns to the organ are counted, so
the large, trivially correct background does not dilute the numbers.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ORGANS = ["esophagus", "heart", "trachea", "aorta"]


def load(folder: Path, name: str) -> np.ndarray:
    return np.array(Image.open(folder / name)) // 63


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_dir", type=Path)
    parser.add_argument("runs", type=Path, nargs="+", help="run folders of the same config, one per seed")
    parser.add_argument("--img-dir", type=Path, default=None, help="CT slices, only needed for --figure")
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    names = sorted(p.name for p in args.gt_dir.glob("*.png"))
    # per organ: [errors where seeds agree, errors where they disagree,
    #             voxels where they agree, voxels where they disagree]
    counts = np.zeros((len(ORGANS), 4))
    most_uncertain = (0, None)

    for name in names:
        gt = load(args.gt_dir, name)
        preds = np.stack([load(r / "best_epoch" / "val", name) for r in args.runs])
        disagree = (preds != preds[0]).any(axis=0)
        if disagree.sum() > most_uncertain[0]:
            most_uncertain = (int(disagree.sum()), name)

        for k in range(1, 5):
            relevant = (gt == k) | (preds == k).any(axis=0)
            errors = ((preds == k) != (gt == k)) & relevant  # per seed
            counts[k - 1] += [errors[:, ~disagree].sum(), errors[:, disagree].sum(),
                              (relevant & ~disagree).sum() * len(args.runs),
                              (relevant & disagree).sum() * len(args.runs)]

    print(f"{len(names)} slices, {len(args.runs)} seeds; voxels where any seed or the label has the organ\n")
    print(f"{'organ':10s} {'disagreeing area':>17s} {'errors there':>13s} "
          f"{'error rate there':>17s} {'error rate elsewhere':>21s}")
    for k, organ in enumerate(ORGANS):
        err_agree, err_dis, n_agree, n_dis = counts[k]
        print(f"{organ:10s} {n_dis / (n_agree + n_dis):16.1%} {err_dis / max(err_agree + err_dis, 1):12.1%} "
              f"{err_dis / max(n_dis, 1):16.1%} {err_agree / max(n_agree, 1):20.1%}")

    if args.figure and args.img_dir:
        import matplotlib.pyplot as plt
        name = most_uncertain[1]
        img = np.array(Image.open(args.img_dir / name))
        img = img[..., 0] if img.ndim == 3 else img
        gt = load(args.gt_dir, name)
        preds = np.stack([load(r / "best_epoch" / "val", name) for r in args.runs])
        disagree = (preds != preds[0]).any(axis=0)
        wrong = (preds != gt).any(axis=0)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        for ax, (title, overlay) in zip(axes, [("CT slice", None), ("Ground truth", gt),
                                               ("Seed 1 prediction", preds[0]),
                                               ("Seeds disagree (red), any seed wrong (blue)", None)]):
            ax.imshow(img, cmap="gray")
            if overlay is not None:
                ax.imshow(np.ma.masked_equal(overlay, 0), cmap="tab10", vmin=0, vmax=9, alpha=0.6)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        axes[3].imshow(np.ma.masked_where(~wrong, wrong), cmap="Blues", vmin=0, vmax=1.5, alpha=0.6)
        axes[3].imshow(np.ma.masked_where(~disagree, disagree), cmap="Reds", vmin=0, vmax=1.2, alpha=0.6)
        fig.suptitle(f"Where the three seeds disagree is where the model makes its mistakes ({name})")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=150)
        print(f"\nSaved {args.figure} (slice with the most disagreement)")


if __name__ == "__main__":
    main()

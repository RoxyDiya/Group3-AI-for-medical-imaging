"""Collect every finished run into one Excel file.

    python analysis/build_results_table.py runs results.xlsx

Sheets:
    runs        one row per run: config, seed, CO2 and per organ 3D metrics
    summary     mean and std over seeds per config and organ, and the change
                against the baseline, with what was changed
    per_patient every value per patient, for paired tests later

A run is any folder holding run_config.txt and metrics/. To add your runs,
copy metrics/*.npz, run_config.txt and emissions.csv of each run into
runs/<run name>/. The config name is the
folder name without its _seedN suffix. The baseline is the config with variant
baseline, uniform sampling, ce loss, no augmentation, 1 context slice, ENet
with base width 8, Adam at lr 0.0005, no scheduler and 2D epoch selection.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["dice_3d", "hd95_3d", "assd_3d", "cldice_3d"]
ORGANS = ["esophagus", "heart", "trachea", "aorta"]
SETTINGS = ["variant", "sampling", "loss", "augment", "context", "network", "kernels",
            "optimizer", "lr", "scheduler", "select"]
BASELINE = {"variant": "baseline", "sampling": "uniform", "loss": "ce",
            "augment": "noaug", "context": "1", "network": "enet", "kernels": "8",
            "optimizer": "adam", "lr": "0.0005", "scheduler": "none", "select": "2d"}


def read_config(run: Path) -> dict:
    cfg = {}
    for line in (run / "run_config.txt").read_text().splitlines():
        key, _, value = line.partition(":")
        cfg[key.strip()] = value.strip()
    return cfg


def read_emissions(run: Path) -> dict:
    path = run / "emissions.csv"
    if not path.exists():
        return {}
    last = list(csv.DictReader(path.open()))[-1]  # codecarbon appends, keep the latest
    return {"hours": float(last["duration"]) / 3600,
            "energy_kwh": float(last["energy_consumed"]),
            "co2_kg": float(last["emissions"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    runs, patients = [], []
    for run in sorted(p for p in args.results_dir.iterdir() if (p / "metrics").is_dir()):
        cfg = read_config(run)
        row = {"run": run.name,
               "config": run.name.rsplit("_seed", 1)[0],
               **{k: cfg.get(k, BASELINE[k]) for k in SETTINGS},
               "seed": int(cfg["seed"]),
               "git_commit": cfg.get("git_commit", "unknown"),
               **read_emissions(run)}

        for metric in METRICS:
            data = np.load(run / "metrics" / f"{metric}.npz", allow_pickle=True)
            classes = [str(c) for c in data["classes"]]
            for organ in ORGANS:
                values = data["metric"][:, classes.index(organ)].astype(float)
                if np.isnan(values).all():
                    continue  # clDice is only defined for the tubular organs
                row[f"{metric}_{organ}"] = np.nanmean(values)
                patients += [{"run": run.name, "config": row["config"], "seed": row["seed"],
                              "patient": str(p), "organ": organ, "metric": metric, "value": v}
                             for p, v in zip(data["patients"], values)]
        runs.append(row)

    runs = pd.DataFrame(runs)
    patients = pd.DataFrame(patients)

    # summary: one row per config and organ, mean and std over seeds
    per_run = patients.groupby(["config", "seed", "organ", "metric"])["value"].mean()
    stats = per_run.groupby(["config", "organ", "metric"]).agg(["mean", "std", "count"]).unstack("metric")
    stats.columns = [f"{metric}_{stat}" for stat, metric in stats.columns]

    settings = runs.groupby("config")[SETTINGS].first()
    base = settings[(settings[SETTINGS] == pd.Series(BASELINE)).all(axis=1)].index
    summary = stats.reset_index()
    summary["changed_vs_baseline"] = summary["config"].map(
        lambda c: ", ".join(f"{k}: {BASELINE[k]} -> {settings.loc[c, k]}"
                            for k in SETTINGS if settings.loc[c, k] != BASELINE[k]) or "baseline")
    if len(base) == 1:
        ref = summary[summary["config"] == base[0]].set_index("organ")
        for metric in METRICS:
            col = f"{metric}_mean"
            if col in summary:
                summary[f"{metric}_delta"] = summary[col] - summary["organ"].map(ref[col])

    order = ["config", "changed_vs_baseline", "organ"]
    summary = summary[order + sorted(c for c in summary.columns if c not in order)]
    summary = summary.drop(columns=[c for c in summary.columns if c.endswith("_count")][1:])
    summary = summary.rename(columns={c: "n_seeds" for c in summary.columns if c.endswith("_count")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.out) as writer:
        runs.round(4).to_excel(writer, sheet_name="runs", index=False)
        summary.round(4).to_excel(writer, sheet_name="summary", index=False)
        patients.round(4).to_excel(writer, sheet_name="per_patient", index=False)
    print(f"Wrote {len(runs)} runs, {runs['config'].nunique()} configs to {args.out}")


if __name__ == "__main__":
    main()

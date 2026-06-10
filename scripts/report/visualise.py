# --- Repository path bootstrap ---
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "vendor", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# ---------------------------------
def plot_obj_c_loss_val_loss(logs: dict, out_dir: str, dpi: int):
    """Plot train and val loss for each Obj C run on the same axes."""
    present = {k: logs[k] for k in STRATEGY_LABELS if k in logs}
    if not present:
        print("  [Obj C] No data â€” skipping train/val loss plot")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Objective C â€” Train and Val Loss per Strategy", fontweight='bold')
    for key, (label, colour_key) in STRATEGY_LABELS.items():
        if key in present:
            recs = present[key]
            eps = epochs(recs)
            ax.plot(eps, field(recs, 'train_loss'), color=COLOURS[colour_key], linestyle='-', marker='o', ms=3, label=f"{label} (Train)")
            ax.plot(eps, field(recs, 'val_loss'), color=COLOURS[colour_key], linestyle='--', marker='s', ms=3, alpha=0.7, label=f"{label} (Val)")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train and Validation Loss by Strategy')
    ax.legend(fontsize=8, ncol=2)
    _save(fig, os.path.join(out_dir, 'objC_loss_val_loss.png'), dpi)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualise.py  â€”  FYP Report Visualisation
==========================================
Loads all objective experiment logs from outputs/logs/, exports CSVs, and
generates publication-quality figures for the report.

Handles partial results gracefully â€” only plots data for completed experiments.

Output (written to outputs/report_outputs/figures/):
  results_combined.csv      â€” all epoch records across all objectives
  objA_results.csv
  objB_results.csv
  objC_results.csv
  objD_results.csv

  objA_loss_acc.png         â€” Obj A: train/val loss + accuracy curves
  objA_cer_language.png     â€” Obj A: per-language CER over epochs
  objB_cer_comparison.png   â€” Obj B: TF vs SS val CER + sampling probability
  objB_cer_language.png     â€” Obj B: per-language CER bar chart (final epoch)
  objC_cer_strategy.png     â€” Obj C: val CER per strategy over epochs
  objC_loss_strategy.png    â€” Obj C: train loss per strategy over epochs
  objC_cer_language.png     â€” Obj C: per-language CER bar chart (final epoch)
  objD_cer_pretrain.png     â€” Obj D: val CER over epochs (LRS2 vs CMLR)
  objD_cer_language_bar.png â€” Obj D: per-language CER bar chart (final epoch)

Usage:
  python visualise.py
  python visualise.py --log_dir outputs/logs --out_dir outputs/report_figures
  python visualise.py --dpi 150    # faster preview
"""

import os
import json
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless â€” no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# â”€â”€ Colour palette (colour-blind-friendly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
COLOURS = {
    "objA_second_run":    "#CC3311",   # red-orange
    "english":        "#0077BB",   # blue
    "mandarin":       "#EE7733",   # orange
    "mixed":          "#009988",   # teal
    "overall":        "#333333",   # near-black
    "train":          "#0077BB",
    "val":            "#EE7733",
    "tf":             "#0077BB",   # teacher-forcing
    "ss":             "#EE7733",   # scheduled sampling
    "full_freeze_fixed":     "#009988",   # teal
    "full_freeze_cosine":    "#0077BB",   # blue
    "partial_freeze_fixed":  "#EE7733",   # orange
    "partial_freeze_cosine": "#CC3311",   # red-orange
    "lrs2":           "#0077BB",
    "cmlr":           "#EE7733",
}

STYLE = {
    "figure.dpi":          150,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.linestyle":      "--",
    "grid.alpha":          0.4,
    "font.size":           11,
    "legend.fontsize":     10,
    "axes.titlesize":      13,
    "axes.labelsize":      11,
}
plt.rcParams.update(STYLE)


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_args():
    p = argparse.ArgumentParser(description="VSR FYP Result Visualiser")
    p.add_argument('--log_dir', type=str, default='outputs/logs',
                   help='Directory containing objX_results.json files')
    p.add_argument('--out_dir', type=str, default='outputs/report_figures',
                   help='Output directory for CSVs and PNG figures')
    p.add_argument('--dpi',     type=int, default=300,
                   help='Figure DPI (300 for print, 150 for preview)')
    return p.parse_args()


# â”€â”€ Log Loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

LOG_FILES = {
    "objA_first_run":          "objA_results.json",
    "objA_second_run":    "objA_retry.json",
    "objB_tf":       "objB_tf.json",
    "objB_ss":       "objB_ss.json",
    "objC_full_freeze_fixed":     "objC_full_freeze_fixed.json",
    "objC_full_freeze_cosine":    "objC_full_freeze_cosine.json",
    "objC_full_freeze_plateau":   "objC_full_freeze_plateau.json",
    "objC_partial_freeze_fixed":  "objC_partial_freeze_fixed.json",
    "objC_partial_freeze_cosine": "objC_partial_freeze_cosine.json",
    "objC_partial_freeze_plateau": "objC_partial_freeze_plateau.json",
    "objD_lrs2":     "objD_lrs2.json",
    "objD_cmlr":     "objD_cmlr.json",
}


def load_logs(log_dir: str) -> dict:
    """Load all available JSON logs; skip missing files with a warning."""
    logs = {}
    for key, fname in LOG_FILES.items():
        path = os.path.join(log_dir, fname)
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                logs[key] = data
                print(f"  Loaded  {fname} â€” {len(data)} epoch(s)")
            else:
                print(f"  Empty   {fname} â€” skipping")
        else:
            print(f"  Missing {fname}")
    return logs


def field(records, key, default=float('nan')):
    """Extract a numeric field from a list of epoch dicts."""
    return [r.get(key, default) for r in records]


def epochs(records):
    return [r.get('epoch', i + 1) for i, r in enumerate(records)]


# â”€â”€ CSV Export â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

COMMON_FIELDS = [
    "epoch", "objective", "label",
    "train_loss", "train_acc",
    "val_loss", "val_acc", "val_cer",
    "cer_english", "cer_mandarin", "cer_mixed",
    "n_english", "n_mandarin", "n_mixed", "n_total",
]


def records_to_rows(key: str, records: list) -> list:
    """Normalise an epoch record to a flat CSV row."""
    rows = []
    for r in records:
        # Determine a human-readable label for the series
        if "pretrain_label" in r:
            label = r["pretrain_label"]
        elif "freeze" in r:
            label = f"{r['freeze']}_{r.get('lr_schedule', 'cosine')}"
        elif "mode" in r:
            label = r["mode"]
        else:
            label = key

        obj = key.split("_")[0]   # e.g. "objA", "objB", "objC", "objD"
        row = {"objective": obj, "label": label}
        for f in COMMON_FIELDS:
            if f not in ("objective", "label"):
                row[f] = r.get(f, "")
        rows.append(row)
    return rows


def export_csvs(logs: dict, out_dir: str):
    all_rows = []
    obj_rows = {}

    for key, records in logs.items():
        rows = records_to_rows(key, records)
        all_rows.extend(rows)
        obj = key.split("_")[0]
        obj_rows.setdefault(obj, []).extend(rows)

    # Combined
    combined_path = os.path.join(out_dir, "results_combined.csv")
    _write_csv(combined_path, all_rows)
    print(f"  CSV  â†’ {combined_path}")

    # Per-objective
    for obj, rows in obj_rows.items():
        path = os.path.join(out_dir, f"{obj}_results.csv")
        _write_csv(path, rows)
        print(f"  CSV  â†’ {path}")


def _write_csv(path: str, rows: list):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COMMON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# â”€â”€ Shared Plot Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _save(fig, path: str, dpi: int):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  PNG  â†’ {path}")


def _lang_bar(ax, groups: list, eng: list, man: list, mix: list,
              title: str, x_label: str = "Group"):
    """Side-by-side bar chart for English / Mandarin / Mixed CER per group."""
    x      = np.arange(len(groups))
    width  = 0.25
    bars_e = [v if not np.isnan(v) else 0 for v in eng]
    bars_m = [v if not np.isnan(v) else 0 for v in man]
    bars_x = [v if not np.isnan(v) else 0 for v in mix]

    ax.bar(x - width, bars_e, width, label='English',  color=COLOURS['english'])
    ax.bar(x,         bars_m, width, label='Mandarin', color=COLOURS['mandarin'])
    ax.bar(x + width, bars_x, width, label='Mixed',    color=COLOURS['mixed'])

    # Annotate NaN bars with "N/A"
    for i, v in enumerate(eng):
        if np.isnan(v):
            ax.text(x[i] - width, 0.01, 'N/A', ha='center', va='bottom',
                    fontsize=8, color='grey')
    for i, v in enumerate(man):
        if np.isnan(v):
            ax.text(x[i], 0.01, 'N/A', ha='center', va='bottom',
                    fontsize=8, color='grey')
    for i, v in enumerate(mix):
        if np.isnan(v):
            ax.text(x[i] + width, 0.01, 'N/A', ha='center', va='bottom',
                    fontsize=8, color='grey')

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel(x_label)
    ax.set_ylabel('CER')
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))


# â”€â”€ Objective A â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_obj_a(logs: dict, out_dir: str, dpi: int):
    if "objA_first_run" not in logs and "objA_second_run" not in logs:
        print("  [Obj A] No data â€” skipping")
        return
    recs       = logs.get("objA_first_run", [])
    recs_retry = logs.get("objA_second_run", [])
    eps        = epochs(recs)
    eps_retry  = epochs(recs_retry)

    # â€” Loss & Accuracy â€”
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Objective A â€” Bilingual Fine-tuning: Training Curves", fontweight='bold')

    if recs:
        ax1.plot(eps, field(recs, 'train_loss'), color=COLOURS['train'],
                 marker='o', ms=3, label='Train Loss')
        ax1.plot(eps, field(recs, 'val_loss'),   color=COLOURS['val'],
                 marker='s', ms=3, label='Val Loss', linestyle='--')
    if recs_retry:
        ax1.plot(eps_retry, field(recs_retry, 'train_loss'), color=COLOURS['objA_second_run'],
                 marker='o', ms=3, label='Train Loss (second run)')
        ax1.plot(eps_retry, field(recs_retry, 'val_loss'),   color=COLOURS['objA_second_run'],
                 marker='s', ms=3, label='Val Loss (second run)', linestyle='--', alpha=0.6)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend(fontsize=8)

    if recs:
        ax2.plot(eps, field(recs, 'train_acc'), color=COLOURS['train'],
                 marker='o', ms=3, label='Train Acc')
        ax2.plot(eps, field(recs, 'val_acc'),   color=COLOURS['val'],
                 marker='s', ms=3, label='Val Acc', linestyle='--')
    if recs_retry:
        ax2.plot(eps_retry, field(recs_retry, 'train_acc'), color=COLOURS['objA_second_run'],
                 marker='o', ms=3, label='Train Acc (second run)')
        ax2.plot(eps_retry, field(recs_retry, 'val_acc'),   color=COLOURS['objA_second_run'],
                 marker='s', ms=3, label='Val Acc (second run)', linestyle='--', alpha=0.6)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax2.set_title('Token Accuracy'); ax2.legend(fontsize=8)

    _save(fig, os.path.join(out_dir, 'objA_loss_acc.png'), dpi)

    # â€” Per-language CER â€”
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Objective A â€” Per-language CER over Epochs", fontweight='bold')

    if recs:
        ax.plot(eps, field(recs, 'val_cer'),      color=COLOURS['overall'],
                marker='o', ms=3, label='Overall CER',  lw=2)
        ax.plot(eps, field(recs, 'cer_english'),  color=COLOURS['english'],
                marker='s', ms=3, label='English',  linestyle='--')
        ax.plot(eps, field(recs, 'cer_mandarin'), color=COLOURS['mandarin'],
                marker='^', ms=3, label='Mandarin', linestyle='--')
        ax.plot(eps, field(recs, 'cer_mixed'),    color=COLOURS['mixed'],
                marker='D', ms=3, label='Mixed',    linestyle='--')
    if recs_retry:
        ax.plot(eps_retry, field(recs_retry, 'val_cer'),      color=COLOURS['objA_second_run'],
                marker='o', ms=3, label='Overall CER (second run)', lw=2)
        ax.plot(eps_retry, field(recs_retry, 'cer_english'),  color=COLOURS['objA_second_run'],
                marker='s', ms=3, label='English (second run)',  linestyle='--', alpha=0.6)
        ax.plot(eps_retry, field(recs_retry, 'cer_mandarin'), color=COLOURS['objA_second_run'],
                marker='^', ms=3, label='Mandarin (second run)', linestyle=':', alpha=0.6)
        ax.plot(eps_retry, field(recs_retry, 'cer_mixed'),    color=COLOURS['objA_second_run'],
                marker='D', ms=3, label='Mixed (second run)',    linestyle='-.', alpha=0.6)

    ax.set_xlabel('Epoch'); ax.set_ylabel('CER')
    ax.legend(fontsize=8)
    _save(fig, os.path.join(out_dir, 'objA_cer_language.png'), dpi)


# â”€â”€ Objective B â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_obj_b(logs: dict, out_dir: str, dpi: int):
    tf_recs = logs.get("objB_tf")
    ss_recs = logs.get("objB_ss")

    if not tf_recs and not ss_recs:
        print("  [Obj B] No data â€” skipping")
        return

    # â€” Val CER comparison + sampling probability â€”
    n_rows = 1 + (1 if ss_recs else 0)
    fig, axes = plt.subplots(n_rows, 1 if n_rows == 1 else 1,
                              figsize=(9, 4 + 3 * (n_rows - 1)))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle("Objective B â€” Teacher-Forcing vs Scheduled Sampling", fontweight='bold')

    ax0 = axes[0]
    if tf_recs:
        ax0.plot(epochs(tf_recs), field(tf_recs, 'val_cer'),
                 color=COLOURS['tf'], marker='o', ms=3, label='Teacher-Forcing')
    if ss_recs:
        ax0.plot(epochs(ss_recs), field(ss_recs, 'val_cer'),
                 color=COLOURS['ss'], marker='s', ms=3, label='Scheduled Sampling',
                 linestyle='--')
    ax0.set_xlabel('Epoch'); ax0.set_ylabel('Val CER')
    ax0.set_title('Validation CER Comparison'); ax0.legend()

    if ss_recs and n_rows > 1:
        ax1 = axes[1]
        ax1.plot(epochs(ss_recs), field(ss_recs, 'sampling_prob', 0.0),
                 color=COLOURS['ss'], marker='.', ms=4)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Sampling Probability')
        ax1.set_title('Scheduled Sampling Probability Schedule')

    _save(fig, os.path.join(out_dir, 'objB_cer_comparison.png'), dpi)

    # â€” Per-language CER bar chart (final epoch each) â€”
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Objective B â€” Per-language CER (Final Epoch)", fontweight='bold')

    groups = []
    eng_vals, man_vals, mix_vals = [], [], []
    for label, recs in [("Teacher-Forcing", tf_recs), ("Scheduled Sampling", ss_recs)]:
        if recs:
            last = recs[-1]
            groups.append(label)
            eng_vals.append(last.get('cer_english',  float('nan')))
            man_vals.append(last.get('cer_mandarin', float('nan')))
            mix_vals.append(last.get('cer_mixed',    float('nan')))

    if groups:
        _lang_bar(ax, groups, eng_vals, man_vals, mix_vals,
                  title='Per-language CER (Final Epoch)', x_label='Training Method')

    _save(fig, os.path.join(out_dir, 'objB_cer_language.png'), dpi)


# â”€â”€ Objective C â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

STRATEGY_LABELS = {
    "objC_full_freeze_fixed":     ("Full Freeze, Fixed LR",       "full_freeze_fixed"),
    "objC_full_freeze_cosine":    ("Full Freeze, Cosine Sched",   "full_freeze_cosine"),
    "objC_full_freeze_plateau":   ("Full Freeze, Plateau",        "full_freeze_fixed"),
    "objC_partial_freeze_fixed":  ("Partial Freeze, Fixed LR",    "partial_freeze_fixed"),
    "objC_partial_freeze_cosine": ("Partial Freeze, Cosine Sched","partial_freeze_cosine"),
    "objC_partial_freeze_plateau":("Partial Freeze, Plateau",     "partial_freeze_fixed"),
}


def plot_obj_c(logs: dict, out_dir: str, dpi: int):
    present = {k: logs[k] for k in STRATEGY_LABELS if k in logs}
    if not present:
        print("  [Obj C] No data â€” skipping")
        return

    # â€” Val CER per strategy â€”
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Objective C â€” Fine-tuning Strategy: Val CER", fontweight='bold')
    for key, (label, colour_key) in STRATEGY_LABELS.items():
        if key in present:
            recs = present[key]
            ax.plot(epochs(recs), field(recs, 'val_cer'),
                    color=COLOURS[colour_key], marker='o', ms=3, label=label)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val CER')
    ax.legend()
    _save(fig, os.path.join(out_dir, 'objC_cer_strategy.png'), dpi)

    # â€” Train Loss per strategy â€”
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Objective C â€” Fine-tuning Strategy: Train Loss", fontweight='bold')
    for key, (label, colour_key) in STRATEGY_LABELS.items():
        if key in present:
            recs = present[key]
            ax.plot(epochs(recs), field(recs, 'train_loss'),
                    color=COLOURS[colour_key], marker='o', ms=3, label=label)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss')
    ax.legend()
    _save(fig, os.path.join(out_dir, 'objC_loss_strategy.png'), dpi)

    # â€” Per-language CER bar chart (final epoch each) â€”
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Objective C â€” Per-language CER by Strategy (Final Epoch)",
                 fontweight='bold')
    groups, eng_vals, man_vals, mix_vals = [], [], [], []
    for key, (label, _) in STRATEGY_LABELS.items():
        if key in present:
            last = present[key][-1]
            groups.append(label)
            eng_vals.append(last.get('cer_english',  float('nan')))
            man_vals.append(last.get('cer_mandarin', float('nan')))
            mix_vals.append(last.get('cer_mixed',    float('nan')))

    if groups:
        _lang_bar(ax, groups, eng_vals, man_vals, mix_vals,
                  title='Per-language CER (Final Epoch)', x_label='Strategy')

    _save(fig, os.path.join(out_dir, 'objC_cer_language.png'), dpi)


# â”€â”€ Objective D â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_obj_d(logs: dict, out_dir: str, dpi: int):
    lrs2_recs = logs.get("objD_lrs2")
    cmlr_recs = logs.get("objD_cmlr")

    if not lrs2_recs and not cmlr_recs:
        print("  [Obj D] No data â€” skipping")
        return

    # â€” Val CER over epochs â€”
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Objective D â€” Pretrain Language Effect: Val CER", fontweight='bold')
    if lrs2_recs:
        ax.plot(epochs(lrs2_recs), field(lrs2_recs, 'val_cer'),
                color=COLOURS['lrs2'], marker='o', ms=3, label='LRS2 (English pretrain)')
    if cmlr_recs:
        ax.plot(epochs(cmlr_recs), field(cmlr_recs, 'val_cer'),
                color=COLOURS['cmlr'], marker='s', ms=3,
                label='CMLR (Mandarin pretrain)', linestyle='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val CER')
    ax.legend()
    _save(fig, os.path.join(out_dir, 'objD_cer_pretrain.png'), dpi)

    # â€” Per-language CER bar chart (final epoch) â€”
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Objective D â€” Per-language CER by Pretrain Source (Final Epoch)",
                 fontweight='bold')
    groups, eng_vals, man_vals, mix_vals = [], [], [], []
    for label, recs in [("LRS2\n(English)", lrs2_recs),
                         ("CMLR\n(Mandarin)", cmlr_recs)]:
        if recs:
            last = recs[-1]
            groups.append(label)
            eng_vals.append(last.get('cer_english',  float('nan')))
            man_vals.append(last.get('cer_mandarin', float('nan')))
            mix_vals.append(last.get('cer_mixed',    float('nan')))

    if groups:
        _lang_bar(ax, groups, eng_vals, man_vals, mix_vals,
                  title='Per-language CER by Pretrain Source (Final Epoch)',
                  x_label='Pretrain Dataset')

    _save(fig, os.path.join(out_dir, 'objD_cer_language_bar.png'), dpi)


# â”€â”€ Cross-Objective Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_summary(logs: dict, out_dir: str, dpi: int):
    """Best final-epoch val CER for each experiment series â€” one overview bar chart."""
    labels, cers = [], []
    series_map = [
        ("objA",             "Obj A\n(orig)"),
        ("objA_retry",       "Obj A\n(retry)"),
        ("objB_tf",          "Obj B\nTF"),
        ("objB_ss",          "Obj B\nSS"),
        ("objC_diff_lr",     "Obj C\nDiff LR"),
        ("objC_full",        "Obj C\nFull FT"),
        ("objC_full_rlrop",  "Obj C\nFull RLROP"),
        ("objC_full_ext",    "Obj C\nFull Ext"),
        ("objC_frozen",      "Obj C\nFrozen"),
        ("objD_lrs2",        "Obj D\nLRS2"),
        ("objD_cmlr",        "Obj D\nCMLR"),
    ]
    for key, label in series_map:
        if key in logs and logs[key]:
            # best CER across all epochs this series
            val_cers = [r.get('val_cer', float('nan')) for r in logs[key]]
            val_cers = [v for v in val_cers if not np.isnan(v)]
            if val_cers:
                labels.append(label)
                cers.append(min(val_cers))

    if len(cers) < 2:
        return   # not enough data for a meaningful summary

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4))
    fig.suptitle("Cross-Objective Summary â€” Best Val CER", fontweight='bold')
    x = np.arange(len(labels))
    bars = ax.bar(x, cers, color="#0077BB", width=0.5)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Best Val CER (â†“ better)')
    ax.set_ylim(0, max(cers) * 1.25)
    _save(fig, os.path.join(out_dir, 'summary_best_cer.png'), dpi)


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nLoading logs from: {args.log_dir}")
    logs = load_logs(args.log_dir)

    if not logs:
        print("No experiment logs found.  Run at least one obj_*_train.py first.")
        return

    print(f"\nExporting CSVs â†’")
    export_csvs(logs, args.out_dir)

    print(f"\nGenerating figures â†’ (DPI={args.dpi})")

    plot_obj_a(logs, args.out_dir, args.dpi)
    plot_obj_b(logs, args.out_dir, args.dpi)
    plot_obj_c(logs, args.out_dir, args.dpi)
    plot_obj_c_loss_val_loss(logs, args.out_dir, args.dpi)
    plot_obj_d(logs, args.out_dir, args.dpi)
    plot_summary(logs, args.out_dir, args.dpi)

    print(f"\nDone.  All outputs written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()





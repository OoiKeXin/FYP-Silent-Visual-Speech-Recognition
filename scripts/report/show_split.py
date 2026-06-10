#!/usr/bin/env python3
# --- Repository path bootstrap ---
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "vendor", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# ---------------------------------
# -*- coding: utf-8 -*-
"""
show_split.py  â€”  Visualise the Train/Val/Test Data Split
==========================================================
Reads the actual Final_Split directory and reports:
  - Per-split totals and language breakdown
  - Token sequence length statistics
  - The stratified-split + train-balancing methodology used

Usage:
  python show_split.py
  python show_split.py --npz_root data/raw/CSLR_Strata/Final_Split
"""

import os
import argparse
import numpy as np

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DEFAULT_ROOT = r"data/raw/CSLR_Strata/Final_Split"
SPLITS = ["train", "val", "test"]
LANG_KEYS = ["eng", "chi", "mix"]
LANG_LABELS = {"eng": "English", "chi": "Mandarin", "mix": "Mixed (code-switch)"}

# â”€â”€ Language detection (matches resplitting.py exactly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def detect_language(text: str) -> str:
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    has_english = any(c.isalpha() and c.isascii() for c in text)
    if has_chinese and has_english:
        return "mix"
    elif has_chinese:
        return "chi"
    elif has_english:
        return "eng"
    return "unknown"


# â”€â”€ Scan split directory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def scan_split(npz_dir: str) -> dict:
    """Returns per-sample metadata for every .npz in the directory."""
    records = []
    for fname in sorted(os.listdir(npz_dir)):
        if not fname.endswith(".npz"):
            continue
        d = np.load(os.path.join(npz_dir, fname), allow_pickle=True)
        label   = str(d["real_label"])
        tok_len = len(d["token_ids"])
        records.append({
            "file":    fname,
            "label":   label,
            "lang":    detect_language(label),
            "tok_len": tok_len,
        })
    return records


# â”€â”€ Pretty printing helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

W = 70

def rule(c="â”€"):  print(c * W)
def header(t):
    pad = (W - len(t) - 2) // 2
    print("â”€" * pad + " " + t + " " + "â”€" * (W - pad - len(t) - 2))


def bar_chart(counts: dict, total: int, width: int = 30) -> str:
    lines = []
    for key in LANG_KEYS:
        n   = counts.get(key, 0)
        pct = 100 * n / total if total else 0
        bar = "â–ˆ" * int(pct * width / 100)
        lines.append(
            f"  {LANG_LABELS[key]:<24} {n:>4}  ({pct:5.1f}%)  {bar}"
        )
    return "\n".join(lines)


def seq_stats(lens: list) -> str:
    if not lens:
        return "  No data"
    return (
        f"  min={min(lens)}  max={max(lens)}  "
        f"mean={sum(lens)/len(lens):.1f}  "
        f"median={sorted(lens)[len(lens)//2]}"
    )


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_args():
    p = argparse.ArgumentParser(description="Show data split distribution")
    p.add_argument("--npz_root", default=DEFAULT_ROOT,
                   help="Root directory containing train/ val/ test/ subfolders")
    return p.parse_args()


def main():
    args = get_args()

    print()
    rule("â•")
    header("Dataset Split  â€”  VSR Bilingual FYP")
    rule("â•")

    # â”€â”€ Methodology recap â”€â”€
    print("""
  Splitting methodology (resplitting.py):
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚ 1. Detect language of every sample from real_label          â”‚
  â”‚    (eng / chi / mix)                                        â”‚
  â”‚                                                             â”‚
  â”‚ 2. Stratified split â€” preserves language ratio in all sets  â”‚
  â”‚    Train 70%  â”‚  Val 15%  â”‚  Test 15%                       â”‚
  â”‚    sklearn train_test_split(stratify=labels, seed=42)       â”‚
  â”‚                                                             â”‚
  â”‚ 3. Balance train set â€” undersample to equal per-class count â”‚
  â”‚    Ensures model sees equal English / Mandarin / Mixed      â”‚
  â”‚    during training (val/test left as-is)                    â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
""")

    # â”€â”€ Scan each split â”€â”€
    all_data = {}
    for split in SPLITS:
        npz_dir = os.path.join(args.npz_root, split, "npz")
        if not os.path.isdir(npz_dir):
            print(f"  [WARN] {npz_dir} not found â€” skipping")
            continue
        print(f"  Scanning {split}...", end="  ", flush=True)
        records = scan_split(npz_dir)
        all_data[split] = records
        print(f"{len(records)} samples loaded")

    print()
    rule("â•")

    # â”€â”€ Per-split detailed stats â”€â”€
    totals = {}
    for split in SPLITS:
        if split not in all_data:
            continue
        records = all_data[split]
        counts  = {k: sum(1 for r in records if r["lang"] == k) for k in LANG_KEYS}
        total   = len(records)
        totals[split] = total
        lens    = [r["tok_len"] for r in records]

        header(f"{split.upper()}  â€”  {total} samples")
        print()
        print(bar_chart(counts, total))
        print()
        print(f"  Token sequence lengths:")
        print(seq_stats(lens))
        print()
        rule()

    # â”€â”€ Cross-split comparison table â”€â”€
    print()
    header("Cross-Split Summary")
    print()

    col_w = 12
    hdr   = f"  {'Language':<24}" + "".join(f"{s.upper():>{col_w}}" for s in SPLITS if s in all_data) + f"{'TOTAL':>{col_w}}"
    print(hdr)
    print("  " + "â”€" * (len(hdr) - 2))

    grand_total = {k: 0 for k in LANG_KEYS}
    for key in LANG_KEYS:
        row = f"  {LANG_LABELS[key]:<24}"
        row_total = 0
        for split in SPLITS:
            if split not in all_data:
                continue
            n = sum(1 for r in all_data[split] if r["lang"] == key)
            row_total += n
            grand_total[key] += n
            row += f"{n:>{col_w}}"
        row += f"{row_total:>{col_w}}"
        print(row)

    # Totals row
    print("  " + "â”€" * (len(hdr) - 2))
    tot_row = f"  {'TOTAL':<24}"
    overall = 0
    for split in SPLITS:
        if split not in all_data:
            continue
        n = len(all_data[split])
        overall += n
        tot_row += f"{n:>{col_w}}"
    tot_row += f"{overall:>{col_w}}"
    print(tot_row)
    print()

    # â”€â”€ Split ratio verification â”€â”€
    header("Split Ratios (% of total)")
    print()
    if overall:
        for split in SPLITS:
            if split not in all_data:
                continue
            n   = len(all_data[split])
            pct = 100 * n / overall
            bar = "â–ˆ" * int(pct / 2)
            print(f"  {split:<8} {n:>5} / {overall}  ({pct:5.1f}%)  {bar}")
    print()

    # â”€â”€ Balance verification â”€â”€
    if "train" in all_data:
        header("Train Set Balance Check")
        print()
        tr = all_data["train"]
        counts = {k: sum(1 for r in tr if r["lang"] == k) for k in LANG_KEYS}
        is_balanced = len(set(counts.values())) == 1
        for key in LANG_KEYS:
            print(f"  {LANG_LABELS[key]:<24}: {counts[key]}")
        status = "âœ… Perfectly balanced" if is_balanced else "âš ï¸  Imbalanced"
        print(f"\n  {status}")
        print()

    rule("â•")
    print()


if __name__ == "__main__":
    main()




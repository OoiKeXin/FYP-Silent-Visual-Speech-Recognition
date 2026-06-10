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
show_normalisation.py  â€”  Visualise the Text Normalisation Pipeline
====================================================================
Shows the step-by-step transformation from raw label to model input tokens
as used in all four objective training scripts.

Pipeline:
  Step 1  raw_label (original text, may be mixed Chinese-English)
          e.g.  "tennisæ¯”èµ›"
          |
  Step 2  Chinese chars â†’ tone-stripped Pinyin romanisation (uppercase)
          English chars â†’ uppercase
          Syllables space-separated
          e.g.  TENNIS  BI SAI
          |
  Step 3  Map each character to CHAR_LIST index â†’ token_id sequence
          e.g.  Tâ†’33  Eâ†’18  Nâ†’27 ...  <space>â†’13 ...

The normalization is pre-computed when the .npz dataset files are created;
at training time the model only ever sees token_ids (integers).

Usage:
  python show_normalisation.py                   # shows 12 samples from val set
  python show_normalisation.py --n 5             # show 5 samples
  python show_normalisation.py --split train     # use train split
  python show_normalisation.py --lang english    # only show English samples
  python show_normalisation.py --lang mandarin   # only show Mandarin samples
  python show_normalisation.py --lang mixed      # only show code-switch samples
"""

import os
import argparse
import numpy as np

# â”€â”€ Vocabulary (identical to all training scripts) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHAR_LIST = [
    "<blank>", "<unk>", "'",
    "0","1","2","3","4","5","6","7","8","9",
    "<space>",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "<eos>"
]
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_LIST)}
EOS_ID  = len(CHAR_LIST) - 1   # 40
PAD_ID  = 0                     # <blank>


# â”€â”€ Language detector (identical to training scripts) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def detect_language(text: str) -> str:
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    has_latin   = any(c.isalpha() and ord(c) < 128 for c in text)
    if has_chinese and has_latin:
        return "mixed"
    elif has_chinese:
        return "mandarin"
    elif has_latin:
        return "english"
    return "unknown"


# â”€â”€ Decode token_ids â†’ readable string (for verification) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def decode_ids(ids) -> str:
    parts = []
    for t in ids:
        if t == PAD_ID or t == EOS_ID:
            continue
        ch = CHAR_LIST[t] if 0 <= t < len(CHAR_LIST) else f"[{t}]"
        parts.append(" " if ch == "<space>" else ch)
    return "".join(parts)


# â”€â”€ Pretty terminal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
W = 72   # line width

def rule(char="â”€"):
    print(char * W)

def header(text):
    pad = (W - len(text) - 2) // 2
    print("â”€" * pad + " " + text + " " + "â”€" * (W - pad - len(text) - 2))


# â”€â”€ Per-sample display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def show_sample(fname: str, data, idx: int):
    real_label = str(data["real_label"])
    token_ids  = data["token_ids"].tolist()
    token_str  = [str(t) for t in data["token_str"]]
    lang       = detect_language(real_label)

    # Reconstruct the normalised text from token_str for display
    normalised = "".join(" " if t == "<space>" else t for t in token_str)

    # Verify: does decoding token_ids match token_str text?
    decoded_from_ids = decode_ids(token_ids)
    consistent = (decoded_from_ids.strip() == normalised.strip())

    header(f"Sample {idx+1}  [{lang.upper()}]  {fname}")

    # â”€â”€ Step 1: Raw label â”€â”€
    print(f"\n  STEP 1 â€” Original label (real_label)")
    print(f"  â”Œ{'â”€'*54}â”")
    print(f"  â”‚  {real_label:<52}â”‚")
    print(f"  â””{'â”€'*54}â”˜")

    # â”€â”€ Step 2: Normalised token string â”€â”€
    print(f"\n  STEP 2 â€” Normalised form (token_str)")
    print(f"           Chinese chars â†’ tone-stripped Pinyin (uppercase)")
    print(f"           English chars â†’ uppercase")
    print(f"           Syllables space-separated")
    print(f"  â”Œ{'â”€'*54}â”")
    print(f"  â”‚  {normalised:<52}â”‚")
    print(f"  â””{'â”€'*54}â”˜")

    # â”€â”€ Step 3: Token IDs â”€â”€
    print(f"\n  STEP 3 â€” Token IDs  (token_ids â†’ CHAR_LIST index)")
    # Show aligned char â†’ id mapping
    chars_line = ""
    ids_line   = ""
    for ch, tid in zip(token_str, token_ids):
        display = ch if ch != "<space>" else "SPC"
        width   = max(len(display), len(str(tid))) + 1
        chars_line += display.center(width)
        ids_line   += str(tid).center(width)

    # Word-wrap at W-4 chars
    chunk = W - 6
    for start in range(0, len(chars_line), chunk):
        print(f"  char  {chars_line[start:start+chunk]}")
        print(f"  id    {ids_line[start:start+chunk]}")
        if start + chunk < len(chars_line):
            print()

    # â”€â”€ Step 4: CHAR_LIST mapping legend â”€â”€
    print(f"\n  STEP 4 â€” Verification (decode token_ids â†’ text)")
    print(f"  â”Œ{'â”€'*54}â”")
    mark = "âœ“" if consistent else "âœ— MISMATCH"
    print(f"  â”‚  {decoded_from_ids:<50}  {mark}â”‚")
    print(f"  â””{'â”€'*54}â”˜")

    print(f"\n  Summary")
    print(f"  {'raw label':<22}: {real_label}")
    print(f"  {'language':<22}: {lang}")
    print(f"  {'sequence length':<22}: {len(token_ids)} tokens")
    print(f"  {'unique tokens used':<22}: {sorted(set(token_ids))}")
    print()


# â”€â”€ Vocab reference table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def print_vocab_table():
    rule("â•")
    header("CHAR_LIST  â€”  Full Vocabulary  (41 tokens, IDs 0â€“40)")
    print()
    cols = 6
    entries = [(i, c) for i, c in enumerate(CHAR_LIST)]
    rows = (len(entries) + cols - 1) // cols
    for r in range(rows):
        row_items = entries[r::rows]
        line = "  ".join(f"{i:>2}: {c:<8}" for i, c in row_items)
        print(f"  {line}")
    print()
    rule("â•")
    print()


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_args():
    p = argparse.ArgumentParser(
        description="Show the text normalisation pipeline used in training"
    )
    p.add_argument("--npz_dir", default=r"data/raw/CSLR_Strata/Final_Split\val\npz",
                   help="Directory with .npz token files")
    p.add_argument("--n",    type=int, default=12,
                   help="Number of samples to display")
    p.add_argument("--split", choices=["train", "val", "test"], default="val",
                   help="Which split to sample from")
    p.add_argument("--lang",  default=None,
                   choices=["english", "mandarin", "mixed"],
                   help="Filter by language (default: show all)")
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()

    # Auto-adjust path if --split is given
    if args.split != "val" and "Final_Split" in args.npz_dir:
        args.npz_dir = args.npz_dir.replace(
            "Final_Split\\val", f"Final_Split\\{args.split}"
        ).replace(
            "Final_Split/val", f"Final_Split/{args.split}"
        )

    if not os.path.isdir(args.npz_dir):
        print(f"[ERROR] Directory not found: {args.npz_dir}")
        return

    all_files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
    if not all_files:
        print(f"[ERROR] No .npz files in {args.npz_dir}")
        return

    # Filter by language if requested
    if args.lang:
        filtered = []
        for fname in all_files:
            d = np.load(os.path.join(args.npz_dir, fname), allow_pickle=True)
            if detect_language(str(d["real_label"])) == args.lang:
                filtered.append(fname)
        all_files = filtered
        if not all_files:
            print(f"[ERROR] No {args.lang} samples found")
            return

    # Sample up to --n files, ensuring coverage of language types if no filter
    import random
    rng = random.Random(args.seed)
    if len(all_files) <= args.n:
        selected = all_files
    elif not args.lang:
        # Stratified: equal English / Mandarin / Mixed
        buckets = {"english": [], "mandarin": [], "mixed": [], "unknown": []}
        for fname in all_files:
            d = np.load(os.path.join(args.npz_dir, fname), allow_pickle=True)
            buckets[detect_language(str(d["real_label"]))].append(fname)
        per_lang = max(1, args.n // 3)
        selected = []
        for lang in ("english", "mandarin", "mixed"):
            selected.extend(rng.sample(buckets[lang],
                                       min(per_lang, len(buckets[lang]))))
        selected = selected[:args.n]
    else:
        selected = rng.sample(all_files, min(args.n, len(all_files)))

    # â”€â”€ Print header â”€â”€
    print()
    rule("â•")
    header("Text Normalisation Pipeline  â€”  VSR Bilingual FYP")
    rule("â•")
    print(f"\n  Source dir : {os.path.abspath(args.npz_dir)}")
    print(f"  Split      : {args.split}")
    print(f"  Filter     : {args.lang or 'all languages'}")
    print(f"  Showing    : {len(selected)} samples\n")

    print_vocab_table()

    for idx, fname in enumerate(selected):
        path = os.path.join(args.npz_dir, fname)
        data = np.load(path, allow_pickle=True)
        show_sample(fname, data, idx)
        rule()
        print()

    # â”€â”€ Summary statistics â”€â”€
    header("Dataset Statistics  (full split)")
    counts = {"english": 0, "mandarin": 0, "mixed": 0, "unknown": 0}
    lens   = []
    all_npz = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
    for fname in all_npz:
        d = np.load(os.path.join(args.npz_dir, fname), allow_pickle=True)
        counts[detect_language(str(d["real_label"]))] += 1
        lens.append(len(d["token_ids"]))
    total = sum(counts.values())
    print(f"\n  Total samples : {total}")
    for lang, n in sorted(counts.items()):
        if n > 0:
            pct = 100 * n / total if total else 0
            bar = "â–ˆ" * int(pct / 2)
            print(f"  {lang:<10}: {n:>4}  ({pct:5.1f}%)  {bar}")
    if lens:
        print(f"\n  Token seq length â€” min {min(lens)}  max {max(lens)}  "
              f"mean {sum(lens)/len(lens):.1f}")
    print()
    rule("â•")
    print()


if __name__ == "__main__":
    main()




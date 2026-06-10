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
show_batching.py  â€”  Visualise Padding & Data Integration Pipeline
===================================================================
Shows exactly what happens between raw data files and the model's encoder input:

  Part A â€” VIDEO PADDING
    Raw .pt files have variable frame counts.  collate_fn zero-pads all videos
    in a batch to the longest T, producing a dense (B, 1, T_max, H, W) tensor.

  Part B â€” TOKEN PADDING
    Raw token_ids have variable lengths.  collate_fn right-pads with PAD_ID=0
    (<blank>) to the longest sequence in the batch.

  Part C â€” DATA INTEGRATION
    Shows how one (video, tokens, label) triplet travels through VSRDataset
    and collate_fn, and what the encoder actually receives.

Outputs (saved to outputs/report_outputs/figures/):
  batching_video_pad.png   â€” frame-count bar chart + padded batch tensor diagram
  batching_token_pad.png   â€” token length bar chart + padded token grid heatmap
  batching_integration.png â€” end-to-end pipeline diagram for one sample

Usage:
  python show_batching.py
  python show_batching.py --batch_size 6 --dpi 150
"""

import os
import argparse
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# â”€â”€ Vocab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHAR_LIST = [
    "<blank>", "<unk>", "'",
    "0","1","2","3","4","5","6","7","8","9",
    "<space>",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "<eos>"
]
EOS_ID = len(CHAR_LIST) - 1   # 40
PAD_ID = 0                     # <blank>

STYLE = {
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "font.size":        10,
}
plt.rcParams.update(STYLE)

BLUE   = "#0077BB"
ORANGE = "#EE7733"
TEAL   = "#009988"
GREY   = "#AAAAAA"
RED    = "#CC3311"


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def detect_language(text):
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    has_latin   = any(c.isalpha() and ord(c) < 128 for c in text)
    if has_chinese and has_latin: return "mixed"
    elif has_chinese:             return "mandarin"
    elif has_latin:               return "english"
    return "unknown"

LANG_COLOUR = {"english": BLUE, "mandarin": ORANGE, "mixed": TEAL, "unknown": GREY}

def load_pair(pt_path, npz_path):
    video  = torch.load(pt_path, weights_only=False).float()
    data   = np.load(npz_path, allow_pickle=True)
    tokens = torch.tensor(data["token_ids"]).long()
    label  = str(data["real_label"])
    tstr   = [str(t) for t in data["token_str"]]
    if tokens[-1] != EOS_ID:
        tokens = torch.cat([tokens, torch.tensor([EOS_ID])])
    # Normalise video to (1, T, H, W)
    if video.ndim == 3:
        video = video.unsqueeze(0)
    if video.shape[0] != 1 and video.shape[1] == 1:
        video = video.permute(1, 0, 2, 3)
    if video.shape[0] != 1:
        video = video.mean(0, keepdim=True)
    return video, tokens, label, tstr

def _save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved â†’ {path}")


# â”€â”€ Part A: Video Padding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_video_padding(samples, out_dir, dpi):
    """Bar chart of raw frame counts + ASCII-style batch tensor diagram."""
    names      = [s["key"] for s in samples]
    frame_cnts = [s["T_raw"] for s in samples]
    T_max      = max(frame_cnts)
    langs      = [s["lang"] for s in samples]
    colours    = [LANG_COLOUR[l] for l in langs]

    fig = plt.figure(figsize=(13, 8))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4)

    # â”€â”€ Subplot 1: Raw frame counts â”€â”€
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(range(len(names)), frame_cnts, color=colours, edgecolor="white", lw=0.5)
    ax1.axhline(T_max, color=RED, linestyle="--", lw=1.5,
                label=f"T_max = {T_max} (batch maximum â†’ pad target)")
    ax1.bar_label(bars, fmt="%d", padding=2, fontsize=8)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([f"{n}\n({l})" for n, l in zip(names, langs)],
                        fontsize=7, rotation=15, ha="right")
    ax1.set_ylabel("Frame count (T)")
    ax1.set_title("Step 1 â€” Raw video frame counts per sample in one batch\n"
                  "(each .pt file has a different number of frames)", fontweight="bold")
    legend_patches = [mpatches.Patch(color=LANG_COLOUR[l], label=l.capitalize())
                      for l in ("english", "mandarin", "mixed")]
    ax1.legend(handles=legend_patches + [
        plt.Line2D([0],[0], color=RED, linestyle="--", label=f"T_max={T_max}")
    ], fontsize=8)

    # â”€â”€ Subplot 2: Padded batch grid â”€â”€
    ax2 = fig.add_subplot(gs[1, 0])
    # Grid: rows = samples, cols = T_max.  Real frames = 1.0, padded = 0.0
    grid = np.zeros((len(samples), T_max))
    for i, s in enumerate(samples):
        grid[i, :s["T_raw"]] = 1.0   # real frames
    im = ax2.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                    interpolation="nearest")
    ax2.set_xlabel("Time step (frame index)")
    ax2.set_ylabel("Sample in batch")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([f"[{i}] {n}" for i, n in enumerate(names)], fontsize=7)
    ax2.set_title(f"Step 2 â€” After collate_fn: padded batch\n"
                  f"Shape = ({len(samples)}, 1, {T_max}, H, W)",
                  fontweight="bold")
    # Colour bar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["pad (0.0)", "real frame"])

    # â”€â”€ Subplot 3: Frame count distribution across full val set â”€â”€
    ax3 = fig.add_subplot(gs[1, 1])
    all_T = [s["T_raw"] for s in samples]   # only batch here; could pass full set
    ax3.hist(all_T, bins=min(10, len(all_T)), color=BLUE, edgecolor="white")
    ax3.set_xlabel("Frame count (T)")
    ax3.set_ylabel("# samples")
    ax3.set_title("Frame count distribution\n(this batch)", fontweight="bold")
    ax3.axvline(np.mean(all_T), color=RED, linestyle="--",
                label=f"mean={np.mean(all_T):.1f}")
    ax3.legend(fontsize=8)

    fig.suptitle("Video Padding Pipeline  â€”  VSRDataset â†’ collate_fn â†’ Encoder Input",
                 fontweight="bold", fontsize=12, y=1.01)

    _save(fig, os.path.join(out_dir, "batching_video_pad.png"), dpi)


# â”€â”€ Part B: Token Padding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_token_padding(samples, out_dir, dpi):
    """Heatmap of padded token_ids batch + length bar chart."""
    token_lens = [s["L_raw"] for s in samples]
    L_max      = max(token_lens)
    names      = [s["key"] for s in samples]
    langs      = [s["lang"] for s in samples]

    # Build padded matrix (values = token_id, padded positions = -1 for colouring)
    matrix = np.full((len(samples), L_max), -1, dtype=float)
    for i, s in enumerate(samples):
        ids = s["token_ids"]
        matrix[i, :len(ids)] = ids

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(samples) * 0.65 + 2)))
    fig.suptitle("Token Padding Pipeline  â€”  collate_fn â†’ Decoder Input",
                 fontweight="bold", fontsize=12)

    # â”€â”€ Heatmap â”€â”€
    ax1 = axes[0]
    # Real tokens: viridis colour by ID (0â€“40); pad = grey
    display = np.where(matrix == -1, np.nan, matrix)
    im = ax1.imshow(display, aspect="auto", cmap="viridis", vmin=0, vmax=40,
                    interpolation="nearest")
    # Overlay pad cells in grey
    pad_mask = matrix == -1
    grey_overlay = np.zeros((*matrix.shape, 4))
    grey_overlay[pad_mask] = [0.85, 0.85, 0.85, 1.0]
    ax1.imshow(grey_overlay, aspect="auto", interpolation="nearest")

    # Annotate cells (only if few enough)
    if L_max <= 30 and len(samples) <= 8:
        for r in range(len(samples)):
            for c in range(L_max):
                val = matrix[r, c]
                if val == -1:
                    ax1.text(c, r, "PAD", ha="center", va="center",
                             fontsize=5, color="#888888")
                else:
                    tid  = int(val)
                    name = CHAR_LIST[tid] if tid < len(CHAR_LIST) else str(tid)
                    ax1.text(c, r, f"{name}\n{tid}", ha="center", va="center",
                             fontsize=4.5, color="white" if tid > 15 else "black")

    cbar = plt.colorbar(im, ax=ax1, fraction=0.03, pad=0.04)
    cbar.set_label("Token ID (0â€“40)", fontsize=8)
    ax1.set_xlabel("Token position")
    ax1.set_ylabel("Sample in batch")
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels([f"[{i}] {n}\n({l})" for i, (n, l)
                         in enumerate(zip(names, langs))], fontsize=7)
    ax1.set_title(f"Padded token_ids batch\n"
                  f"Shape = ({len(samples)}, {L_max})   PAD_ID = 0 (<blank>)",
                  fontweight="bold")

    # Mark real-vs-pad boundary per row with a vertical tick
    for i, s in enumerate(samples):
        ax1.plot(s["L_raw"] - 0.5, i, marker="|", color=RED,
                 markersize=10, markeredgewidth=1.5)

    # â”€â”€ Length bar chart â”€â”€
    ax2 = axes[1]
    bar_colours = [LANG_COLOUR[l] for l in langs]
    bars = ax2.barh(range(len(names)), token_lens, color=bar_colours,
                    edgecolor="white", lw=0.5)
    ax2.axvline(L_max, color=RED, linestyle="--", lw=1.5,
                label=f"L_max = {L_max}")
    ax2.bar_label(bars, fmt="%d", padding=2, fontsize=8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([f"[{i}] {n}\n({l})"
                         for i, (n, l) in enumerate(zip(names, langs))], fontsize=7)
    ax2.set_xlabel("Token sequence length (L)")
    ax2.set_title("Raw token lengths\n(before padding)", fontweight="bold")
    legend_patches = [mpatches.Patch(color=LANG_COLOUR[l], label=l.capitalize())
                      for l in ("english", "mandarin", "mixed")]
    ax2.legend(handles=legend_patches + [
        plt.Line2D([0],[0], color=RED, linestyle="--", label=f"L_max={L_max}")
    ], fontsize=8)
    ax2.invert_yaxis()

    _save(fig, os.path.join(out_dir, "batching_token_pad.png"), dpi)


# â”€â”€ Part C: Integration Diagram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_integration(sample, out_dir, dpi):
    """End-to-end pipeline for one sample: files â†’ dataset â†’ collate â†’ encoder."""
    T, H, W   = sample["T_raw"], sample["H"], sample["W"]
    L         = sample["L_raw"]
    tstr      = sample["token_str"]
    tids      = sample["token_ids"]
    raw_label = sample["label"]
    lang      = sample["lang"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Data Integration Pipeline  â€”  Sample: {sample['key']}  "
        f"[{lang.upper()}]  |  \"{raw_label}\"",
        fontweight="bold", fontsize=11
    )

    # â”€â”€ [0,0] Raw video: sample of frames â”€â”€
    ax = axes[0, 0]
    video = sample["video"]   # (1, T, H, W)
    n_show = min(5, T)
    idxs   = np.linspace(0, T - 1, n_show, dtype=int)
    strip  = np.concatenate([video[0, i].numpy() for i in idxs], axis=1)
    ax.imshow(strip, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"â‘  Raw video (.pt)\nShape: ({T}, 1, {H}, {W})\n"
                 f"{n_show} frames shown (equally spaced)", fontweight="bold")
    ax.set_axis_off()
    for j, idx in enumerate(idxs):
        ax.text(W * j + W // 2, H + 3, f"f{idx}", ha="center",
                va="top", fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.1", fc="#333", alpha=0.7))

    # â”€â”€ [0,1] Raw token_ids â”€â”€
    ax = axes[0, 1]
    ax.axis("off")
    ax.set_title(f"â‘¡ Raw tokens (.npz)\ntoken_ids length = {L}",
                 fontweight="bold")
    # Table of (token_str, token_id) pairs
    col_w = max(2, len(tstr) // 4) if len(tstr) > 0 else 4
    rows  = []
    for start in range(0, len(tstr), col_w * 2):
        chunk = list(zip(tstr[start:start + col_w * 2],
                         tids[start:start + col_w * 2]))
        rows.append(chunk)
    y = 0.95
    ax.text(0.02, y, f"real_label: \"{raw_label}\"", transform=ax.transAxes,
            fontsize=9, va="top", weight="bold")
    y -= 0.12
    ax.text(0.02, y, "token_str  â†’  token_id", transform=ax.transAxes,
            fontsize=8, va="top", color="#555")
    y -= 0.10
    for chunk in rows:
        line = "   ".join(f"{ts:<7} â†’ {tid:>2}" for ts, tid in chunk)
        ax.text(0.02, y, line, transform=ax.transAxes,
                fontsize=7.5, va="top", family="monospace")
        y -= 0.09
        if y < 0.02:
            break

    # â”€â”€ [0,2] VSRDataset.__getitem__ output â”€â”€
    ax = axes[0, 2]
    ax.axis("off")
    ax.set_title("â‘¢ VSRDataset.__getitem__\n(normalised shapes)", fontweight="bold")
    info = [
        ("video tensor",   f"shape: (1, {T}, {H}, {W})"),
        ("dtype",          "torch.float32"),
        ("value range",    f"[{sample['vmin']:.3f}, {sample['vmax']:.3f}]"),
        ("mean / std",     f"{sample['vmean']:.3f} / {sample['vstd']:.3f}"),
        ("",               ""),
        ("token_ids",      f"shape: ({L},)"),
        ("dtype",          "torch.int64"),
        ("starts with",    f"[{', '.join(str(x) for x in tids[:5])}...]"),
        ("ends with",      f"[...{', '.join(str(x) for x in tids[-3:])}]"),
        ("EOS_ID=40?",     "âœ“ Yes" if tids[-1] == EOS_ID else "âœ— Missing!"),
        ("",               ""),
        ("real_label",     f"\"{raw_label}\""),
        ("language",       lang),
    ]
    y = 0.97
    for label, val in info:
        if label == "":
            y -= 0.04
            continue
        ax.text(0.02, y, f"{label:<16}", transform=ax.transAxes,
                fontsize=8, va="top", color="#555", style="italic")
        ax.text(0.42, y, val, transform=ax.transAxes,
                fontsize=8, va="top", weight="bold")
        y -= 0.075

    # â”€â”€ [1,0] Video padding (show how THIS sample sits in a padded batch) â”€â”€
    ax = axes[1, 0]
    T_max = sample["T_max_batch"]
    B     = sample["B"]
    grid  = np.zeros((B, T_max))
    for i, t_i in enumerate(sample["all_T"]):
        grid[i, :t_i] = 1.0
    im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    # Highlight this sample
    this_row = sample["batch_idx"]
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.5, this_row - 0.5), T_max, 1,
        boxstyle="round,pad=0.1", linewidth=2,
        edgecolor=RED, facecolor="none", label="this sample"
    ))
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Batch sample")
    ax.set_title(f"â‘£ Video after collate_fn\n"
                 f"Batch shape: ({B}, 1, {T_max}, {H}, {W})\n"
                 f"Zero-padding: {T_max - T} frames added to this sample",
                 fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["pad", "real"])
    ax.legend(loc="lower right", fontsize=7)

    # â”€â”€ [1,1] Token padding heatmap â”€â”€
    ax = axes[1, 1]
    L_max     = sample["L_max_batch"]
    all_tids  = sample["all_token_ids"]
    tok_grid  = np.full((B, L_max), np.nan)
    for i, ids in enumerate(all_tids):
        tok_grid[i, :len(ids)] = ids
    im2 = ax.imshow(tok_grid, aspect="auto", cmap="viridis", vmin=0, vmax=40)
    grey_ov = np.zeros((*tok_grid.shape, 4))
    grey_ov[np.isnan(tok_grid)] = [0.85, 0.85, 0.85, 1.0]
    ax.imshow(grey_ov, aspect="auto")
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.5, this_row - 0.5), L_max, 1,
        boxstyle="round,pad=0.1", linewidth=2,
        edgecolor=RED, facecolor="none"
    ))
    ax.set_xlabel("Token position")
    ax.set_ylabel("Batch sample")
    ax.set_title(f"â‘¤ Tokens after collate_fn\n"
                 f"Batch shape: ({B}, {L_max})\n"
                 f"PAD_ID=0 appended: {L_max - L} positions",
                 fontweight="bold")
    plt.colorbar(im2, ax=ax, fraction=0.04, pad=0.04).set_label("Token ID")

    # â”€â”€ [1,2] What encoder sees â”€â”€
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_title("â‘¥ What the model receives\n(one batch, ready for forward pass)",
                 fontweight="bold")
    enc_info = [
        ("â”€â”€ VIDEO (encoder input) â”€â”€", ""),
        ("padded_videos", f"({B}, 1, {T_max}, {H}, {W})"),
        ("dtype",         "torch.float32"),
        ("real content",  f"frames 0â€“{T-1}   (T={T})"),
        ("zero padding",  f"frames {T}â€“{T_max-1}   (+{T_max-T} frames)"),
        ("pixel range",   "[0.0, 1.0]  (normalised)"),
        ("",              ""),
        ("â”€â”€ TOKENS (decoder input) â”€â”€", ""),
        ("padded_tokens", f"({B}, {L_max})"),
        ("dtype",         "torch.int64"),
        ("real content",  f"positions 0â€“{L-1}   (L={L})"),
        ("pad content",   f"positions {L}â€“{L_max-1}   (PAD_ID=0)"),
        ("",              ""),
        ("â”€â”€ LABELS array â”€â”€",          ""),
        ("real_labels",   f"list of {B} strings"),
        ("this sample",   f"\"{raw_label}\""),
    ]
    y = 0.97
    for lbl, val in enc_info:
        if "â”€â”€" in lbl:
            y -= 0.02
            ax.text(0.02, y, lbl, transform=ax.transAxes,
                    fontsize=8, va="top", weight="bold", color="#333")
            y -= 0.07
            continue
        if lbl == "":
            y -= 0.04
            continue
        ax.text(0.02, y, f"{lbl:<18}", transform=ax.transAxes,
                fontsize=7.5, va="top", color="#555", style="italic")
        ax.text(0.50, y, val, transform=ax.transAxes,
                fontsize=7.5, va="top", weight="bold")
        y -= 0.067

    _save(fig, os.path.join(out_dir, "batching_integration.png"), dpi)


# â”€â”€ Collect batch samples â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def collect_batch(pt_dir, npz_dir, batch_size, seed):
    pt_keys  = {os.path.splitext(f)[0] for f in os.listdir(pt_dir)  if f.endswith(".pt")}
    npz_keys = {os.path.splitext(f)[0] for f in os.listdir(npz_dir) if f.endswith(".npz")}
    keys     = sorted(pt_keys & npz_keys)

    rng = random.Random(seed)

    # Stratified: try to get mixed languages
    buckets = {"english": [], "mandarin": [], "mixed": []}
    for k in keys:
        d = np.load(os.path.join(npz_dir, k + ".npz"), allow_pickle=True)
        lang = detect_language(str(d["real_label"]))
        if lang in buckets:
            buckets[lang].append(k)

    per  = max(1, batch_size // 3)
    sel  = []
    for lang in ("english", "mandarin", "mixed"):
        sel.extend(rng.sample(buckets[lang], min(per, len(buckets[lang]))))
    # Top up if needed
    remaining = [k for bucket in buckets.values() for k in bucket if k not in sel]
    rng.shuffle(remaining)
    sel = (sel + remaining)[:batch_size]

    samples = []
    for k in sel:
        video, tokens, label, tstr = load_pair(
            os.path.join(pt_dir,  k + ".pt"),
            os.path.join(npz_dir, k + ".npz")
        )
        _, T, H, W = video.shape
        samples.append({
            "key":       k,
            "video":     video,
            "token_ids": tokens.tolist(),
            "token_str": tstr,
            "label":     label,
            "lang":      detect_language(label),
            "T_raw":     T,
            "L_raw":     len(tokens),
            "H": H, "W": W,
            "vmin":  video.min().item(),
            "vmax":  video.max().item(),
            "vmean": video.mean().item(),
            "vstd":  video.std().item(),
        })

    T_max = max(s["T_raw"] for s in samples)
    L_max = max(s["L_raw"] for s in samples)
    all_T = [s["T_raw"] for s in samples]
    all_L = [s["token_ids"] for s in samples]

    for i, s in enumerate(samples):
        s["T_max_batch"] = T_max
        s["L_max_batch"] = L_max
        s["B"]           = len(samples)
        s["all_T"]       = all_T
        s["all_token_ids"] = all_L
        s["batch_idx"]   = i

    return samples


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_args():
    p = argparse.ArgumentParser(description="Visualise padding & data integration pipeline")
    p.add_argument("--pt_dir",    default=r"data/raw/CSLR_Strata/Final_Split\val\pt")
    p.add_argument("--npz_dir",   default=r"data/raw/CSLR_Strata/Final_Split\val\npz")
    p.add_argument("--out_dir",   default="outputs/report_figures")
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--seed",       type=int, default=7)
    p.add_argument("--dpi",        type=int, default=200)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nCollecting batch of {args.batch_size} samples â€¦")
    samples = collect_batch(args.pt_dir, args.npz_dir, args.batch_size, args.seed)

    print(f"\nSample summary:")
    T_max = max(s["T_raw"] for s in samples)
    L_max = max(s["L_raw"] for s in samples)
    for s in samples:
        pad_v = T_max - s["T_raw"]
        pad_t = L_max - s["L_raw"]
        print(f"  {s['key']}  [{s['lang']:<8}]  "
              f"video=({s['T_raw']}â†’{T_max}, +{pad_v} pad)  "
              f"tokens=({s['L_raw']}â†’{L_max}, +{pad_t} pad)  "
              f"\"{s['label']}\"")

    print(f"\nGenerating figures â€¦")
    plot_video_padding(samples, args.out_dir, args.dpi)
    plot_token_padding(samples, args.out_dir, args.dpi)
    # Use the first "mixed" sample for the integration diagram (most interesting)
    focus = next((s for s in samples if s["lang"] == "mixed"), samples[0])
    plot_integration(focus, args.out_dir, args.dpi)

    print(f"\nDone.  Outputs in: {os.path.abspath(args.out_dir)}")
    print("  batching_video_pad.png")
    print("  batching_token_pad.png")
    print("  batching_integration.png")


if __name__ == "__main__":
    main()




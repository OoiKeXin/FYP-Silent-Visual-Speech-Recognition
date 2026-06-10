# --- Repository path bootstrap ---
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "vendor", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# ---------------------------------
"""
Show example frames from an .npz video sample.

Usage:
    python show_frame.py                                        # default sample
    python show_frame.py path/to/file.npz                      # specific file
    python show_frame.py path/to/file.npz --frames 0 10 44 89  # specific frame indices
    python show_frame.py path/to/file.npz --all                # save all 90 frames
"""

import sys
import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

DEFAULT_PATH = r"data/raw/CSLR_Strata/Dataset Part_01\000_000_029.npz"

# Find a CJK-capable font on Windows; fall back gracefully
def _get_cjk_font():
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "NSimSun", "FangSong"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

_CJK_FONT = _get_cjk_font()

def parse_args():
    parser = argparse.ArgumentParser(description="Visualise frames from a .npz video sample.")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH, help="Path to .npz file")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="Frame indices to display (default: evenly spaced 9 frames)")
    parser.add_argument("--all", action="store_true", help="Save all frames as a grid")
    parser.add_argument("--out", default=None, help="Output PNG path (default: auto-named)")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"[ERROR] File not found: {args.path}")
        sys.exit(1)

    data = np.load(args.path, allow_pickle=True)
    video = data["video"]           # (T, H, W) uint8
    real_label = str(data["real_label"])
    video_length = int(data["video_length"])
    label_length = int(data["label_length"])
    T, H, W = video.shape

    print(f"File        : {args.path}")
    print(f"Label       : {real_label}")
    print(f"Video shape : {video.shape}  (T={T}, H={H}, W={W})")
    print(f"video_length: {video_length}")
    print(f"label_length: {label_length}")

    # Determine which frames to show
    if args.all:
        frame_indices = list(range(T))
        ncols = 10
    elif args.frames is not None:
        frame_indices = [f for f in args.frames if 0 <= f < T]
    else:
        # 9 evenly spaced frames
        frame_indices = [int(round(i * (T - 1) / 8)) for i in range(9)]

    nframes = len(frame_indices)
    ncols = min(9, nframes) if not args.all else 10
    nrows = (nframes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.7))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, fi in enumerate(frame_indices):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.imshow(video[fi], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"t={fi}", fontsize=7)
        ax.axis("off")

    # Hide unused axes
    for idx in range(nframes, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    basename = os.path.splitext(os.path.basename(args.path))[0]
    tag = "all" if args.all else f"frames_{'_'.join(map(str, frame_indices))}"
    out_path = args.out or f"{basename}_{tag}.png"

    title = f"{basename}  |  \"{real_label}\"  |  {T} frames"
    suptitle_kwargs = {"fontsize": 9}
    if _CJK_FONT:
        suptitle_kwargs["fontproperties"] = fm.FontProperties(family=_CJK_FONT, size=9)
    fig.suptitle(title, **suptitle_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved â†’ {out_path}")

if __name__ == "__main__":
    main()




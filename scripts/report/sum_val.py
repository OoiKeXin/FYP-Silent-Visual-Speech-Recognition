# --- Repository path bootstrap ---
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "vendor", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# ---------------------------------
import json

# ===== INPUT =====
input_file = "outputs/logs/objA_results.json"   # your 100-epoch log
output_file = "outputs/logs/val_result_ObjA.json"

label = "ObjA"
description = "Obj A"

# ===== LOAD DATA =====
with open(input_file, "r") as f:
    data = json.load(f)

# ===== SAFE AVERAGE FUNCTION =====
def avg(key):
    values = [d[key] for d in data if key in d]
    return sum(values) / len(values) if values else 0

# ===== SAMPLE COUNTS (take from last epoch or max) =====
def get_samples(key):
    values = [d[key] for d in data if key in d]
    return max(values) if values else 0

# ===== BUILD SUMMARY =====
summary = {
    "label": label,
    "description": description,
    "metrics": {
        "overall": {
            "cer": avg("val_cer"),
            "samples": get_samples("n_total"),
        },
        "english": {
            "cer": avg("cer_english"),
            "samples": get_samples("n_english"),
        },
        "mandarin": {
            "cer": avg("cer_mandarin"),
            "samples": get_samples("n_mandarin"),
        },
        "mixed": {
            "cer": avg("cer_mixed"),
            "samples": get_samples("n_mixed"),
        },
    },
}

# ===== SAVE JSON =====
with open(output_file, "w") as f:
    json.dump(summary, f, indent=2)

print("Saved summary to", output_file)



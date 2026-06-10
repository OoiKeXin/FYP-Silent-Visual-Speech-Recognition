# --- Repository path bootstrap ---
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "vendor", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# ---------------------------------
import json
import matplotlib.pyplot as plt

result_files = [
    "outputs/logs/test_result_ObjA_user_best.json",
    "outputs/logs/test_result_ObjC_full_freeze_cosine_user_best.json",
    "outputs/logs/test_result_ObjC_full_freeze_plateau_user_best.json",
    "outputs/logs/test_result_ObjC_partial_freeze_cosine_user_best.json",
    "outputs/logs/test_result_ObjC_partial_freeze_plateau_user_best.json",
]

# Use predefined labels directly
labels = [
    "ObjA - Baseline Fine-tuning",
    "ObjC - Full Freeze (Cosine)",
    "ObjC - Full Freeze (Plateau)",
    "ObjC - Partial Freeze (Cosine)",
    "ObjC - Partial Freeze (Plateau)",
]

overall_cers = []
english_cers = []
mandarin_cers = []
mixed_cers = []

for file in result_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        metrics = data["metrics"]

        overall_cers.append(metrics["overall"]["cer"])
        english_cers.append(metrics["english"]["cer"])
        mandarin_cers.append(metrics["mandarin"]["cer"])
        mixed_cers.append(metrics["mixed"]["cer"])

# Ensure alignment
x = range(len(labels))
width = 0.2

plt.figure(figsize=(12, 6))
plt.bar([i - 1.5*width for i in x], overall_cers, width, label="Overall CER")
plt.bar([i - 0.5*width for i in x], english_cers, width, label="English CER")
plt.bar([i + 0.5*width for i in x], mandarin_cers, width, label="Mandarin CER")
plt.bar([i + 1.5*width for i in x], mixed_cers, width, label="Mixed CER")

plt.xticks(x, labels, rotation=20, ha="right")
plt.ylabel("CER")
plt.title("CER Comparison Across Models")
plt.legend()
plt.tight_layout()
plt.savefig("cer_comparison.png", dpi=200)

print("Plot saved as cer_comparison.png")



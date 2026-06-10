# --- Repository path bootstrap ---
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "vendor", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# ---------------------------------
import pandas as pd

# =========================
# Test Data
# =========================
test_results = [
    {"label": "ObjA", "overall": 0.7939, "english": 0.8281, "mandarin": 0.7502, "mixed": 0.8020},
    {"label": "ObjC Full Cosine", "overall": 0.8052, "english": 0.8312, "mandarin": 0.7771, "mixed": 0.8064},
    {"label": "ObjC Full Plateau", "overall": 0.8127, "english": 0.8323, "mandarin": 0.7854, "mixed": 0.8196},
    {"label": "ObjC Partial Cosine", "overall": 0.7925, "english": 0.8278, "mandarin": 0.7512, "mixed": 0.7972},
    {"label": "ObjC Partial Plateau", "overall": 0.7977, "english": 0.8321, "mandarin": 0.7511, "mixed": 0.8088},
]

# =========================
# Validation Data
# =========================
val_results = [
    {"label": "ObjA", "overall": 0.8266, "english": 0.8502, "mandarin": 0.7982, "mixed": 0.8305},
    {"label": "ObjC Full Cosine", "overall": 0.8141, "english": 0.8399, "mandarin": 0.7844, "mixed": 0.8171},
    {"label": "ObjC Full Plateau", "overall": 0.8386, "english": 0.8586, "mandarin": 0.8163, "mixed": 0.8404},
    {"label": "ObjC Partial Cosine", "overall": 0.8126, "english": 0.8365, "mandarin": 0.7872, "mixed": 0.8135},
    {"label": "ObjC Partial Plateau", "overall": 0.8303, "english": 0.8511, "mandarin": 0.8055, "mixed": 0.8337},
]

# =========================
# Convert to DataFrame
# =========================
df_test = pd.DataFrame(test_results).set_index("label")
df_val = pd.DataFrame(val_results).set_index("label")

# Rename columns
df_test = df_test.add_suffix("_test")
df_val = df_val.add_suffix("_val")

# Combine
df = pd.concat([df_test, df_val], axis=1)

# =========================
# Compute Gap (Val - Test)
# =========================
for col in ["overall", "english", "mandarin", "mixed"]:
    df[f"{col}_gap"] = df[f"{col}_val"] - df[f"{col}_test"]

# =========================
# Round for cleaner display
# =========================
df = df.round(4)

# =========================
# Print Table
# =========================
print("\n=== Test vs Validation CER Table ===\n")
print(df)

# =========================
# Save Outputs
# =========================
df.to_csv("cer_comparison.csv")
df.to_excel("cer_comparison.xlsx")

print("\nSaved to cer_comparison.csv and cer_comparison.xlsx")



# Repository Structure

This repository is organized as a Visual Speech Recognition (VSR) research project for CSLR/Strata experiments. The layout separates source code, runnable scripts, datasets, checkpoints, outputs, references, and vendored upstream code so training and reporting artifacts no longer clutter the project root.

## Top-Level Layout

```text
VSR_Strata/
|-- archive/                 # Legacy scripts and generated Python caches
|-- checkpoints/             # Pretrained and FYP-trained model checkpoints
|-- configs/                 # Hydra configs, model .ini configs, experiment configs
|-- data/                    # Raw, processed, and external benchmark data
|-- outputs/                 # Logs, figures, report figures, and tables
|-- references/              # Papers, original docs, and external setup notes
|-- scripts/                 # Runnable training/evaluation/data/report scripts
|-- src/                     # Project package namespace for future shared code
|-- vendor/                  # Vendored AutoAVSR/ESPnet-style runtime code
|-- LICENSE
|-- README.md
|-- REPOSITORY_STRUCTURE.md
`-- requirements.txt
```

## Folder Purposes

### `scripts/`

Runnable entrypoints live here. These scripts include a repository path bootstrap so they can import `vendor/espnet`, `vendor/pipelines`, and future modules under `src/`.

```text
scripts/
|-- train/       # Objective A-E training experiments
|-- eval/        # Test, inference, benchmark, and mouth-cropping entrypoints
|-- data/        # Dataset inspection, splitting, and movement utilities
`-- report/      # Plotting, visualisation, and report-figure generation
```

Key commands:

```shell
python scripts/train/obj_a_train.py
python scripts/train/obj_c_train.py
python scripts/eval/test_objectives.py
python scripts/report/visualise.py
python scripts/report/show_split.py
```

### `src/vsr_strata/`

Reserved for reusable project code. The current migration creates the package skeleton but keeps behavior in the existing scripts to avoid changing experiment logic during the restructure.

```text
src/vsr_strata/
|-- data/
|-- decoding/
|-- evaluation/
|-- models/
|-- reporting/
`-- training/
```

Recommended next refactor: move repeated dataset/model/decode/CER logic from the objective scripts into this package.

### `vendor/`

Vendored upstream code required by the original AutoAVSR-style inference and ESPnet model runtime.

```text
vendor/
|-- espnet/
`-- pipelines/
```

Imports such as `from espnet...` and `from pipelines...` are preserved by script path bootstrapping.

### `configs/`

Configuration files are grouped by purpose.

```text
configs/
|-- hydra/       # Hydra defaults used by inference/eval/crop scripts
|-- model_ini/   # Original model zoo .ini files
`-- experiments/ # Reserved for future FYP experiment configs
```

### `data/`

Dataset files are separated from source code.

```text
data/
|-- raw/         # Local CSLR_Strata dataset and generated splits
|-- processed/   # Reserved for future processed/intermediate datasets
`-- external/    # External benchmark metadata/models/language-model folders
```

Current CSLR split counts:

| Split | `.pt` files | `.npz` files |
| --- | ---: | ---: |
| Train | 1818 | 1818 |
| Val | 396 | 396 |
| Test | 397 | 397 |

### `checkpoints/`

Model weights are separated from scripts.

```text
checkpoints/
|-- pretrained/
|   `-- LRS2_V_WER26.1/
|-- fyp/
|   |-- obj_a/
|   `-- obj_c/
`-- legacy/
```

### `outputs/`

Generated outputs are no longer stored at the repository root.

```text
outputs/
|-- figures/
|-- logs/
|-- report_figures/
`-- tables/
```

### `references/`

Reference papers, original documentation assets, and original tool setup notes live here.

```text
references/
|-- doc/
|-- papers/
`-- tools/
```

### `archive/`

Older experiments and cache files are preserved rather than deleted.

```text
archive/
|-- cache/
|-- debug_scripts/
`-- legacy_training/
```

## Major File Moves

| Old location | New location |
| --- | --- |
| `obj_a_train.py`, `obj_b_train.py`, `obj_c_train.py`, `obj_d_train.py`, `obj_e_train.py` | `scripts/train/` |
| `test_objectives.py`, `eval.py`, `infer.py`, `crop_mouth.py` | `scripts/eval/` |
| `check.py`, `check_each_output.py`, `check_npz.py`, `resplitting.py`, `moving.py`, `move_pt.py` | `scripts/data/` |
| `visualise.py`, `show_*.py`, `graph.py`, `plot_cer_results.py`, `png.py`, `sum_val.py` | `scripts/report/` |
| `train.py`, `baseline_train.py`, `teacher_train.py`, `greedy_train.py`, `beam_train.py`, `autogressive_train.py`, `test.py` | `archive/legacy_training/` |
| `espnet/`, `pipelines/` | `vendor/` |
| `CSLR_Strata/` | `data/raw/CSLR_Strata/` |
| `benchmarks/` | `data/external/benchmarks/` |
| `LRS2_V_WER26.1/` | `checkpoints/pretrained/LRS2_V_WER26.1/` |
| root `*.pth` checkpoints | `checkpoints/fyp/obj_a/` or `checkpoints/fyp/obj_c/` |
| `experiment_logs/` | `outputs/logs/` |
| `figures/` | `outputs/figures/` |
| `report_figures/` | `outputs/report_figures/` |
| root PDFs | `references/papers/` |
| `doc/`, `tools/` | `references/doc/`, `references/tools/` |

## Duplicate or Legacy Candidates

The following files are preserved in `archive/legacy_training/` because they duplicate logic now covered by objective scripts:

- `train.py`
- `baseline_train.py`
- `teacher_train.py`
- `greedy_train.py`
- `beam_train.py`
- `autogressive_train.py`
- `test.py`

The following cache files are preserved under `archive/cache/` and can be safely regenerated:

- Python `__pycache__/` files
- `.pyc` files

## Verification Notes

The structural migration was verified by checking:

- Key script paths exist under `scripts/`.
- Vendored imports exist under `vendor/`.
- Dataset split folders exist under `data/raw/CSLR_Strata/Final_Split`.
- Train/val/test `.pt` and `.npz` counts remain paired.
- Checkpoints exist under `checkpoints/`.
- Logs and report outputs exist under `outputs/`.
- No stale doubled migration path prefixes remain.

Full Python runtime verification is pending because the current shell resolves `python` to the Microsoft Store stub and no `pip`, `py`, or `conda` executable is available on `PATH`.

## Recommended Next Step

Create or activate a Python environment, install dependencies, then run:

```shell
python scripts/eval/test_objectives.py --help
python scripts/report/show_split.py
python scripts/report/visualise.py
```

For training:

```shell
python scripts/train/obj_a_train.py
python scripts/train/obj_c_train.py --freeze partial_freeze --lr_schedule cosine
```


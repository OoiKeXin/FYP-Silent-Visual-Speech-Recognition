#! /usr/bin/env python
# --- Repository path bootstrap ---
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "vendor", _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# ---------------------------------
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import hydra
from pipelines.pipeline import InferencePipeline


@hydra.main(version_base=None, config_path="../../configs/hydra", config_name="default")
def main(cfg):
    device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu")
    output = InferencePipeline(cfg.config_filename, device=device, detector=cfg.detector, face_track=True)(cfg.data_filename, cfg.landmarks_filename)
    print(f"hyp: {output}")


if __name__ == '__main__':
    main()





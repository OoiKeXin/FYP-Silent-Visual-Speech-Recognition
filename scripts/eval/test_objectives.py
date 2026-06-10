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
test_objectives.py â€” Unified test evaluation for all VSR Objectives (A, B, C, D)
==================================================================================
Loads each objective's best checkpoint and evaluates it on the held-out test set
using greedy decoding.  Reports:
  â€¢ Overall CER
  â€¢ Per-language CER: English / Mandarin / Mixed

Missing checkpoints are skipped gracefully with a warning.

Usage:
    python test_objectives.py
    python test_objectives.py --device cpu
    python test_objectives.py --output_json outputs/logs/test_results.json
    python test_objectives.py --test_pt_dir  data/raw/CSLR_Strata/Final_Split/test/pt \\
                               --test_npz_dir data/raw/CSLR_Strata/Final_Split/test/npz
"""

import os
import json
import argparse

import numpy as np
import torch
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# â”€â”€ Vocabulary (shared across all objectives) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

CHAR_LIST = [
    "<blank>", "<unk>", "'", "0","1","2","3","4","5","6","7","8","9",
    "<space>",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "<eos>"
]
VOCAB_SIZE = len(CHAR_LIST)
EOS_ID     = VOCAB_SIZE - 1
PAD_ID     = CHAR_LIST.index("<blank>")


# â”€â”€ Model Registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each entry: (label, checkpoint_path, description)
# Checkpoints that do not exist will be skipped with a warning.

MODEL_REGISTRY = [
    # â”€â”€ Objective A â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    (
        "ObjA (original)",
        "vsr_objA_best.pth",
        "Obj A: Pinyin bridge â€” LRS2 pretrain, 30 ep, CosineAnnealingLR"
    ),
    (
        "ObjA (retry, RLROP)",
        "vsr_objA_retry_best.pth",
        "Obj A retry: lr=1e-5, ReduceLROnPlateau, 50 ep"
    ),

    # â”€â”€ Objective B â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    (
        "ObjB (teacher-forcing)",
        "vsr_b_tf_best.pth",
        "Obj B: Teacher-forcing baseline"
    ),
    (
        "ObjB (scheduled-sampling)",
        "vsr_b_ss_best.pth",
        "Obj B: Scheduled sampling â€” linear annealing TFâ†’autoregressive"
    ),

    # â”€â”€ Objective C â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    (
        "ObjC (differential LR)",
        "vsr_c_diff_lr_best.pth",
        "Obj C: Differential LR (encÃ—0.1), CosineAnnealingLR"
    ),
    (
        "ObjC (full finetune)",
        "vsr_c_full_best.pth",
        "Obj C: Full fine-tune, CosineAnnealingLR, 30 ep"
    ),
    (
        "ObjC (full, RLROP)",
        "vsr_c_full_rlrop_best.pth",
        "Obj C: Full fine-tune + ReduceLROnPlateau"
    ),
    (
        "ObjC (full, cosine+ext)",
        "vsr_c_full_ext_best.pth",
        "Obj C: Full fine-tune, Cosine+extended epochs"
    ),
    (
        "ObjC (frozen encoder)",
        "vsr_c_frozen_best.pth",
        "Obj C: Frozen encoder â€” decoder-only fine-tune"
    ),

    # â”€â”€ User Best Checkpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ("ObjC (partial freeze, cosine, user best)", "vsr_c_partial_freeze_cosine_best.pth", "Obj C: Partial freeze, Cosine LR, user best checkpoint"),
    ("ObjC (partial freeze, plateau, user best)", "vsr_c_partial_freeze_plateau_best.pth", "Obj C: Partial freeze, Plateau LR, user best checkpoint"),
    ("ObjC (full freeze, cosine, user best)", "vsr_c_full_freeze_cosine_best.pth", "Obj C: Full freeze, Cosine LR, user best checkpoint"),
    ("ObjC (full freeze, plateau, user best)", "vsr_c_full_freeze_plateau_best.pth", "Obj C: Full freeze, Plateau LR, user best checkpoint"),
    ("ObjA (user best)", "vsr_objA_best.pth", "Obj A: User best checkpoint"),

    # â”€â”€ Objective D â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    (
        "ObjD (LRS2 pretrain)",
        "vsr_d_lrs2_best.pth",
        "Obj D: English (LRS2) pretrained encoder â†’ CSLR fine-tune"
    ),
    (
        "ObjD (CMLR pretrain)",
        "vsr_d_cmlr_best.pth",
        "Obj D: Mandarin (CMLR) pretrained encoder â†’ CSLR fine-tune"
    ),
]


# â”€â”€ Language Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def detect_language(real_label: str) -> str:
    if not real_label:
        return "unknown"
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in real_label)
    has_latin   = any(c.isalpha() and ord(c) < 128 for c in real_label)
    if has_chinese and has_latin:
        return "mixed"
    elif has_chinese:
        return "mandarin"
    else:
        return "english"


# â”€â”€ Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class VSRTestDataset(Dataset):
    def __init__(self, pt_dir: str, npz_dir: str):
        self.pt_map  = {os.path.splitext(f)[0]: os.path.join(pt_dir,  f)
                        for f in os.listdir(pt_dir)  if f.endswith(".pt")}
        self.npz_map = {os.path.splitext(f)[0]: os.path.join(npz_dir, f)
                        for f in os.listdir(npz_dir) if f.endswith(".npz")}
        self.keys = sorted(set(self.pt_map) & set(self.npz_map))
        if len(self.keys) == 0:
            raise RuntimeError(f"No matching .pt/.npz pairs found in:\n  {pt_dir}\n  {npz_dir}")
        print(f"Test set: {len(self.keys)} samples")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key  = self.keys[idx]
        data = np.load(self.npz_map[key], allow_pickle=True)
        tokens     = torch.tensor(data["token_ids"]).long()
        real_label = str(data["real_label"]) if "real_label" in data else ""
        if tokens[-1] != EOS_ID:
            tokens = torch.cat([tokens, torch.tensor([EOS_ID])])
        video = torch.load(self.pt_map[key]).float()
        if video.ndim == 3:
            video = video.unsqueeze(0)
        if video.shape[0] != 1 and video.shape[1] == 1:
            video = video.permute(1, 0, 2, 3)
        if video.shape[0] != 1:
            video = video.mean(dim=0, keepdim=True)
        return video, tokens, real_label


def collate_fn(batch):
    videos, tokens, real_labels = zip(*batch)
    max_T = max(v.shape[1] for v in videos)
    max_L = max(t.shape[0] for t in tokens)
    padded_videos = torch.zeros(len(videos), 1, max_T,
                                videos[0].shape[2], videos[0].shape[3])
    padded_tokens = torch.full((len(tokens), max_L), PAD_ID, dtype=torch.long)
    for i, (v, t) in enumerate(zip(videos, tokens)):
        padded_videos[i, :, :v.shape[1]] = v
        padded_tokens[i, :t.shape[0]] = t
    return padded_videos, padded_tokens, list(real_labels)


# â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_model(device: str) -> E2E:
    """Shared Conformer architecture used by all objectives."""
    model_args = Namespace(
        adim=256, aheads=4, elayers=12, eunits=2048,
        dlayers=6, dunits=2048,
        dropout_rate=0.1, transformer_attn_dropout_rate=0.1,
        transformer_input_layer='conv3d',
        transformer_encoder_attn_layer_type='rel_mha',
        macaron_style=False, use_cnn_module=False, cnn_module_kernel=31,
        a_upsample_ratio=1, relu_type='swish', normalization='layernorm',
        mtlalpha=0.3, lsm_weight=0.0,
        transformer_length_normalized_loss=False,
        ctc_type="warpctc", report_cer=False, report_wer=False,
        char_list=CHAR_LIST, sym_blank="<blank>", sym_space="<space>",
    )
    model = E2E(VOCAB_SIZE, model_args).to(device)
    model.criterion = LabelSmoothingLoss(
        size=VOCAB_SIZE, padding_idx=PAD_ID, smoothing=0.05, normalize_length=True,
    )
    return model


# â”€â”€ Token Utilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def tokens_to_text(tokens) -> str:
    text = []
    for t in tokens:
        if t in (PAD_ID, EOS_ID):
            continue
        if t < 0 or t >= len(CHAR_LIST):
            text.append("?")
            continue
        ch = CHAR_LIST[t]
        if ch == "<space>":
            text.append(" ")
        elif ch not in ("<blank>", "<unk>"):
            text.append(ch)
    return "".join(text)


def compute_cer(pred_tokens: list, ref_tokens: list) -> float:
    """Identical to obj_a_train.py: Levenshtein editops on decoded text strings."""
    from Levenshtein import editops
    pred_text = tokens_to_text(pred_tokens)
    ref_text  = tokens_to_text(ref_tokens)
    if len(ref_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0
    ops = editops(ref_text, pred_text)
    S = sum(1 for op, _, _ in ops if op == 'replace')
    D = sum(1 for op, _, _ in ops if op == 'delete')
    I = sum(1 for op, _, _ in ops if op == 'insert')
    C = len(ref_text) - S - D
    denom = S + D + I + C
    return (S + D + I) / denom if denom > 0 else 0.0


# â”€â”€ Greedy Decoding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def greedy_decode(model: E2E, enc_out: torch.Tensor,
                  enc_mask: torch.Tensor, max_len: int = None) -> tuple:
    """Identical to obj_a_train.py greedy_decode. Returns (token_list, eos_found)."""
    device   = enc_out.device
    enc_len  = enc_out.size(1)
    if max_len is None:
        max_len = max(10, min(enc_len * 2, 50))
    ys        = torch.tensor([[model.sos]], device=device)
    ngram_sz  = 3
    found_eos = False
    for _ in range(max_len):
        dec_out, _ = model.decoder(ys, None, enc_out, enc_mask)
        logprobs   = torch.log_softmax(dec_out[:, -1, :], dim=-1)
        if ys.size(1) > 1:
            logprobs[0, ys[0, -1].item()] -= 1.5
        if ys.size(1) > ngram_sz:
            hist   = ys[0].tolist()
            seen   = {tuple(hist[i:i + ngram_sz])
                      for i in range(1, len(hist) - ngram_sz + 1)}
            prefix = tuple(hist[-(ngram_sz - 1):])
            for cand in range(logprobs.size(1)):
                if prefix + (cand,) in seen:
                    logprobs[0, cand] -= 100.0
        next_tok = logprobs.argmax(dim=-1)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        if next_tok.item() == EOS_ID:
            found_eos = True
            break
    return ys[0, 1:].cpu().tolist(), found_eos


# â”€â”€ Evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def evaluate_model(model: E2E, loader: DataLoader, device: str) -> dict:
    """Run greedy decode on every test sample. Return per-language CER stats.
    Uses the same per-sample CER averaging as the objective training scripts."""
    model.eval()
    cer_buckets   = {"english": [], "mandarin": [], "mixed": [], "unknown": []}
    eos_truncated = 0

    with torch.no_grad():
        for videos, tokens, real_labels in tqdm(loader, desc="  decoding", leave=False):
            videos = videos.to(device)
            tokens = tokens.to(device)

            enc_out, enc_mask = model.encoder(videos, None)

            for i in range(videos.size(0)):
                enc_out_i  = enc_out[i:i+1]
                enc_mask_i = enc_mask[i:i+1] if enc_mask is not None else None

                pred_tokens, eos_found = greedy_decode(model, enc_out_i, enc_mask_i)
                if not eos_found:
                    eos_truncated += 1

                ref_tokens = tokens[i].cpu().tolist()
                pred_clean = [t for t in pred_tokens if t not in (PAD_ID, model.sos)]
                ref_clean  = [t for t in ref_tokens  if t not in (PAD_ID, model.sos)]

                cer  = compute_cer(pred_clean, ref_clean)
                lang = detect_language(real_labels[i])
                if lang in cer_buckets:
                    cer_buckets[lang].append(cer)

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    n_total = sum(len(v) for v in cer_buckets.values())
    if eos_truncated > 0:
        print(f"    [decode] EOS not reached: {eos_truncated}/{n_total} samples")

    all_cer = [c for v in cer_buckets.values() for c in v]
    results = {
        "overall":  {"cer": round(safe_mean(all_cer), 4),                     "samples": n_total},
        "english":  {"cer": round(safe_mean(cer_buckets["english"]),  4),     "samples": len(cer_buckets["english"])},
        "mandarin": {"cer": round(safe_mean(cer_buckets["mandarin"]), 4),     "samples": len(cer_buckets["mandarin"])},
        "mixed":    {"cer": round(safe_mean(cer_buckets["mixed"]),    4),     "samples": len(cer_buckets["mixed"])},
    }
    return results


# â”€â”€ Reporting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def print_table(all_results: list):
    col_w = 30
    header = f"{'Model':<{col_w}} {'Overall CER':>12} {'English CER':>12} {'Mandarin CER':>13} {'Mixed CER':>10}"
    print("\n" + "=" * len(header))
    print("  TEST SET RESULTS â€” ALL OBJECTIVES")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, metrics, _ in all_results:
        ov  = metrics.get("overall",  {}).get("cer", float("nan"))
        en  = metrics.get("english",  {}).get("cer", float("nan"))
        mn  = metrics.get("mandarin", {}).get("cer", float("nan"))
        mx  = metrics.get("mixed",    {}).get("cer", float("nan"))
        fmt = lambda v: f"{v:.4f}" if v == v else "  â€”  "
        print(f"  {label:<{col_w-2}} {fmt(ov):>12} {fmt(en):>12} {fmt(mn):>13} {fmt(mx):>10}")
    print("=" * len(header))
    print()


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all VSR objective checkpoints on the test set"
    )
    parser.add_argument('--test_pt_dir',  type=str,
                        default=r"data/raw/CSLR_Strata/Final_Split\test\pt",
                        help="Directory of test .pt video tensors")
    parser.add_argument('--test_npz_dir', type=str,
                        default=r"data/raw/CSLR_Strata/Final_Split\test\npz",
                        help="Directory of test .npz label files")
    parser.add_argument('--batch_size',   type=int,   default=1)
    parser.add_argument('--num_workers',  type=int,   default=0)
    parser.add_argument('--device',       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--output_json',  type=str,
                        default="outputs/logs/test_results.json",
                        help="Path to save JSON results")
    return parser.parse_args()


def main():
    args = get_args()
    print(f"\nDevice : {args.device}")
    print(f"Test PT : {args.test_pt_dir}")
    print(f"Test NPZ: {args.test_npz_dir}\n")

    # â”€â”€ Load test dataset once â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    dataset = VSRTestDataset(args.test_pt_dir, args.test_npz_dir)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # â”€â”€ Build model shell once (weights swapped per checkpoint) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    model = build_model(args.device)


    all_results = []  # list of (label, metrics_dict, description)
    skipped     = []

    for label, ckpt_path, description in MODEL_REGISTRY:
        if not os.path.isfile(ckpt_path):
            print(f"  [SKIP] {label:40s} â€” checkpoint not found: {ckpt_path}")
            skipped.append(label)
            continue

        print(f"\n{'â”€'*60}")
        print(f"  Evaluating: {label}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  {description}")

        sd = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(sd, strict=True)

        metrics = evaluate_model(model, loader, args.device)
        all_results.append((label, metrics, description))

        ov = metrics["overall"]["cer"]
        en = metrics["english"]["cer"]
        mn = metrics["mandarin"]["cer"]
        mx = metrics["mixed"]["cer"]
        print(f"  â†’ Overall CER={ov:.4f} | English={en:.4f} | "
              f"Mandarin={mn:.4f} | Mixed={mx:.4f} "
              f"(n={metrics['overall']['samples']})")

        # Save per-checkpoint result
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("/", "_").replace("-", "_")
        out_path = os.path.join(os.path.dirname(args.output_json), f"test_result_{safe_label}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"label": label, "description": description, "metrics": metrics}, f, indent=2, ensure_ascii=False)
        print(f"  [Saved result to {out_path}]")

    # â”€â”€ Print summary table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if all_results:
        print_table(all_results)

    if skipped:
        print(f"Skipped {len(skipped)} checkpoint(s) (not yet trained):")
        for s in skipped:
            print(f"  â€¢ {s}")
        print()

    # â”€â”€ Save JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    output = {
        label: {"description": desc, "metrics": metrics}
        for label, metrics, desc in all_results
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()




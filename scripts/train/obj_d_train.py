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
Objective D: Pretraining Language Comparison (LRS2 vs CMLR)
=============================================================
Investigates how the language of the pretrained encoder affects adaptation to
bilingual code-switching VSR.  Two encoders are compared:

  LRS2  (English)  â€” trained on BBC Lip Reading Sentences 2 (English, WER 26.1%)
  CMLR  (Mandarin) â€” trained on Baidu CMLR (Mandarin, CER 8.0%)

Both are fine-tuned identically on the bilingual CSLR dataset using the same:
  - Pinyin unified vocabulary
  - Joint CTC + Attention loss (50/50), teacher-forcing
  - Differential LR (encoder Ã— 0.1)
  - CosineAnnealingLR, AMP, gradient clipping, early stopping

Per-language CER (English / Mandarin / Mixed) is the primary metric, allowing
direct measurement of whether the encoder's source language provides an
inductive bias that benefits its corresponding language subset at test time.

This directly addresses the gap in Zhang et al. (2024), which trains a
monolingual CTC model without examining pretraining language effects.

Usage:
  # Run with LRS2 pretrained encoder
  python obj_d_train.py \\
      --pretrain_encoder checkpoints/pretrained/LRS2_V_WER26.1/model.pth \\
      --pretrain_label   lrs2 \\
      --best_model_path  vsr_d_lrs2_best.pth \\
      --experiment_log   outputs/logs/objD_lrs2.json

  # Run with CMLR pretrained encoder (download first â€” see README)
  python obj_d_train.py \\
      --pretrain_encoder data/external/benchmarks/CMLR/models/CMLR_V_WER8.0/model.pth \\
      --pretrain_label   cmlr \\
      --best_model_path  vsr_d_cmlr_best.pth \\
      --experiment_log   outputs/logs/objD_cmlr.json

  # Resume from checkpoint
  python obj_d_train.py --resume vsr_d_lrs2_best.pth --pretrain_label lrs2

NOTE: The CMLR model must be downloaded separately.
  GoogleDrive: https://bit.ly/3fR8RkU  (key: 7eq1)
  Place at:    data/external/benchmarks/CMLR/models/CMLR_V_WER8.0/model.pth
"""

import os
import json
import random

import numpy as np
import torch
from argparse import ArgumentParser, Namespace
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# â”€â”€ Argument Parsing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_args():
    parser = ArgumentParser(
        description="Objective D: LRS2 vs CMLR pretraining effect on per-language CER"
    )

    # Pretrain selection
    parser.add_argument('--pretrain_encoder', type=str,
                        default=r"checkpoints/pretrained/LRS2_V_WER26.1/model.pth",
                        help="Path to pretrained model.pth (LRS2 or CMLR).")
    parser.add_argument('--pretrain_label', type=str, default='lrs2',
                        choices=['lrs2', 'cmlr'],
                        help="Short label recorded in experiment log for comparison.")

    # Paths
    parser.add_argument('--train_video_dir', type=str, default=r"data/raw/CSLR_Strata/Final_Split\train\pt")
    parser.add_argument('--train_token_dir', type=str, default=r"data/raw/CSLR_Strata/Final_Split\train\npz")
    parser.add_argument('--val_video_dir',   type=str, default=r"data/raw/CSLR_Strata/Final_Split\val\pt")
    parser.add_argument('--val_token_dir',   type=str, default=r"data/raw/CSLR_Strata/Final_Split\val\npz")
    parser.add_argument('--resume',          type=str, default=None)
    parser.add_argument('--best_model_path', type=str, default="vsr_d_best.pth")
    parser.add_argument('--experiment_log',  type=str,
                        default="outputs/logs/objD_results.json")

    # Training (identical settings for fair comparison)
    parser.add_argument('--epochs',                  type=int,   default=30)
    parser.add_argument('--batch_size',              type=int,   default=4)
    parser.add_argument('--lr',                      type=float, default=3e-5)
    parser.add_argument('--grad_clip',               type=float, default=5.0)
    parser.add_argument('--seed',                    type=int,   default=42)
    parser.add_argument('--num_workers',             type=int,   default=0)
    parser.add_argument('--amp',                     action='store_true', default=True)
    parser.add_argument('--early_stopping_patience', type=int,   default=5)
    parser.add_argument('--device', type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


# â”€â”€ Vocabulary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Language Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def detect_language(real_label: str) -> str:
    """Classify a sample as 'english', 'mandarin', or 'mixed' from its raw label."""
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

class VSRDataset(Dataset):
    def __init__(self, pt_dir: str, npz_dir: str):
        self.pt_map  = {os.path.splitext(f)[0]: os.path.join(pt_dir,  f)
                        for f in os.listdir(pt_dir)  if f.endswith(".pt")}
        self.npz_map = {os.path.splitext(f)[0]: os.path.join(npz_dir, f)
                        for f in os.listdir(npz_dir) if f.endswith(".npz")}
        self.keys = sorted(set(self.pt_map) & set(self.npz_map))
        assert len(self.keys) > 0, "No matching .pt/.npz pairs found"

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

def build_model(args) -> E2E:
    """Architecture is fixed regardless of pretraining source so that the only
    variable between LRS2 and CMLR runs is the initialised encoder weights."""
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
    model = E2E(VOCAB_SIZE, model_args).to(args.device)
    model.criterion = LabelSmoothingLoss(
        size=VOCAB_SIZE, padding_idx=PAD_ID, smoothing=0.05, normalize_length=True,
    )
    return model


def load_pretrained(model: E2E, path: str, device: str, label: str):
    """Load pretrained weights with shape-matched layer filtering.

    Both LRS2 and CMLR checkpoints share the same Conformer architecture, so
    the encoder layers will always match.  The CTC output and any decoder
    vocabulary projection may differ in size and will be skipped automatically.
    """
    print(f"Loading [{label.upper()}] pretrained weights from: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {path}\n"
            f"For CMLR: download from https://bit.ly/3fR8RkU (key: 7eq1) "
            f"and place at data/external/benchmarks/CMLR/models/CMLR_V_WER8.0/model.pth"
        )
    sd = torch.load(path, map_location=device)
    md = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in md and md[k].shape == v.shape}
    md.update(matched)
    model.load_state_dict(md, strict=False)
    skipped = len(sd) - len(matched)
    print(f"  Matched {len(matched)}/{len(sd)} layers  "
          f"({skipped} skipped due to shape mismatch â€” expected for vocab/output layers)")


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


def compute_cer(pred_tokens, ref_tokens) -> float:
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
                  enc_mask: torch.Tensor, max_len: int = None):
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
        ys       = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        if next_tok.item() == EOS_ID:
            found_eos = True
            break
    if not found_eos:
        print(f"[decode] EOS not found within max_len={max_len}")
    return ys[0, 1:].cpu().tolist()


# â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_one_epoch(model: E2E, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    args, scaler=None, epoch: int = 0) -> tuple:
    """Teacher-forcing training with differential LR.  Identical for both
    LRS2 and CMLR runs â€” the only difference is the initialised weights."""
    model.train()
    total_loss = total_tokens = total_correct = 0
    use_amp = scaler is not None

    for videos, tokens, _ in tqdm(loader, desc=f"  train ep{epoch+1}"):
        videos = videos.to(args.device)
        tokens = tokens.to(args.device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            enc_out, enc_mask = model.encoder(videos, None)
            input_lengths = torch.full(
                (videos.size(0),), enc_out.size(1),
                dtype=torch.long, device=args.device
            )
            ctc_loss = model.ctc(enc_out, input_lengths, tokens)

            sos    = torch.full((tokens.size(0), 1), model.sos,
                                dtype=tokens.dtype, device=args.device)
            ys_in  = torch.cat([sos, tokens[:, :-1]], dim=1)
            ys_out = tokens

            dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
            att_loss   = model.criterion(dec_out, ys_out)
            loss       = 0.5 * ctc_loss + 0.5 * att_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        mask    = ys_out != PAD_ID
        preds   = dec_out.argmax(dim=2)
        correct = (preds == ys_out) & mask

        total_loss    += loss.item() * mask.sum().item()
        total_correct += correct.sum().item()
        total_tokens  += mask.sum().item()

    return total_loss / total_tokens, total_correct / total_tokens


# â”€â”€ Validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def validate(model: E2E, loader: DataLoader, args) -> dict:
    """Evaluate with per-language CER breakdown.

    This is the key metric for Objective D: does an LRS2-pretrained encoder
    produce lower CER on English samples than a CMLR-pretrained encoder, and
    vice versa for Mandarin?  The mixed-language CER reveals cross-lingual
    generalisation ability.
    """
    model.eval()
    total_loss = total_tokens = total_correct = 0
    cer_buckets = {"english": [], "mandarin": [], "mixed": [], "unknown": []}
    eos_truncated = 0

    with torch.no_grad():
        for videos, tokens, real_labels in tqdm(loader, desc="  val"):
            videos = videos.to(args.device)
            tokens = tokens.to(args.device)

            enc_out, enc_mask = model.encoder(videos, None)
            if enc_mask is None:
                enc_mask = torch.ones(
                    enc_out.size(0), 1, enc_out.size(1),
                    device=args.device, dtype=torch.bool
                )
            input_lengths = torch.full(
                (videos.size(0),), enc_out.size(1),
                dtype=torch.long, device=args.device
            )
            ctc_loss = model.ctc(enc_out, input_lengths, tokens)

            sos    = torch.full((tokens.size(0), 1), model.sos,
                                dtype=tokens.dtype, device=tokens.device)
            ys_in  = torch.cat([sos, tokens[:, :-1]], dim=1)
            ys_out = tokens

            dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
            att_loss   = model.criterion(dec_out, ys_out)

            loss    = 0.5 * ctc_loss + 0.5 * att_loss
            mask    = ys_out != PAD_ID
            preds   = dec_out.argmax(dim=2)
            correct = (preds == ys_out) & mask

            total_loss    += loss.item() * mask.sum().item()
            total_correct += correct.sum().item()
            total_tokens  += mask.sum().item()

            for i in range(videos.size(0)):
                pred_tokens, eos_found = greedy_decode(model, enc_out[i:i+1], enc_mask[i:i+1])
                if not eos_found:
                    eos_truncated += 1
                ref_tokens  = tokens[i].cpu().tolist()
                pred_clean  = [t for t in pred_tokens if t not in (PAD_ID, model.sos)]
                ref_clean   = [t for t in ref_tokens  if t not in (PAD_ID, model.sos)]
                cer  = compute_cer(pred_clean, ref_clean)
                lang = detect_language(real_labels[i])
                cer_buckets[lang].append(cer)

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    n_total = sum(len(v) for v in cer_buckets.values())
    if eos_truncated > 0:
        print(f"  [decode] EOS not reached: {eos_truncated}/{n_total} samples (normal in early epochs)")
    all_cer = [c for v in cer_buckets.values() for c in v]
    return {
        "val_loss":     total_loss / total_tokens,
        "val_acc":      total_correct / total_tokens,
        "val_cer":      safe_mean(all_cer),
        "cer_english":  safe_mean(cer_buckets["english"]),
        "cer_mandarin": safe_mean(cer_buckets["mandarin"]),
        "cer_mixed":    safe_mean(cer_buckets["mixed"]),
        "n_english":    len(cer_buckets["english"]),
        "n_mandarin":   len(cer_buckets["mandarin"]),
        "n_mixed":      len(cer_buckets["mixed"]),
        "n_total":      sum(len(v) for v in cer_buckets.values()),
    }


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print(f"Objective D  â€”  pretrain: {args.pretrain_label.upper()}")
    print(f"Device  : {args.device}")
    print(f"Pretrain: {args.pretrain_encoder}")
    print("=" * 60)

    # â”€â”€ Data â”€â”€
    train_set = VSRDataset(args.train_video_dir, args.train_token_dir)
    val_set   = VSRDataset(args.val_video_dir,   args.val_token_dir)
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers
    )

    # â”€â”€ Model â”€â”€
    model = build_model(args)
    if args.resume:
        print(f"Resuming from: {args.resume}")
        sd = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(sd, strict=False)
    else:
        load_pretrained(model, args.pretrain_encoder, args.device, args.pretrain_label)

    # â”€â”€ Differential LR optimizer (same for both runs) â”€â”€
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}
    param_groups = [
        {'params': [p for p in model.parameters() if id(p) in encoder_param_ids],
         'lr': args.lr * 0.1},
        {'params': [p for p in model.parameters() if id(p) not in encoder_param_ids],
         'lr': args.lr},
    ]
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler    = torch.cuda.amp.GradScaler() if args.amp else None

    os.makedirs(os.path.dirname(args.experiment_log), exist_ok=True)

    best_cer   = float('inf')
    no_improve = 0
    results_log = []

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}  [{args.pretrain_label.upper()}]  "
              f"lr_enc={optimizer.param_groups[0]['lr']:.2e}  "
              f"lr_dec={optimizer.param_groups[1]['lr']:.2e}")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, args, scaler, epoch=epoch
        )
        val_m = validate(model, val_loader, args)

        print(f"\nTrain  Loss {tr_loss:.4f} | Acc {tr_acc:.4f}")
        print(f"Val    Loss {val_m['val_loss']:.4f} | Acc {val_m['val_acc']:.4f} "
              f"| CER {val_m['val_cer']:.4f}")
        print("Per-language CER:")
        for lang in ("english", "mandarin", "mixed"):
            n   = val_m[f'n_{lang}']
            cer = val_m[f'cer_{lang}']
            tag = f"{cer:.4f}" if n > 0 else "N/A"
            print(f"  {lang.capitalize():>10}: {tag}  (n={n})")

        results_log.append({
            "epoch":          epoch + 1,
            "pretrain_label": args.pretrain_label,
            "pretrain_path":  args.pretrain_encoder,
            "train_loss":     tr_loss,
            "train_acc":      tr_acc,
            **val_m,
        })

        val_cer = val_m['val_cer']
        if val_cer < best_cer:
            best_cer   = val_cer
            no_improve = 0
            torch.save(model.state_dict(), args.best_model_path)
            print(f"âœ… Best CER {best_cer:.4f} â†’ {args.best_model_path}")
        else:
            no_improve += 1
            print(f"âš ï¸  No improvement for {no_improve} epoch(s) (best={best_cer:.4f})")

        scheduler.step()

        if no_improve >= args.early_stopping_patience:
            print("ðŸ›‘ Early stopping triggered")
            break

        torch.save(model.state_dict(),
                   f"vsr_d_{args.pretrain_label}_epoch{epoch+1}.pth")

    with open(args.experiment_log, 'w') as f:
        json.dump(results_log, f, indent=2)
    print(f"\nResults saved to {args.experiment_log}")


if __name__ == "__main__":
    main()




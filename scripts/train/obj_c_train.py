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
Objective C: Fine-Tuning Strategy Optimisation
===============================================
Empirically compares four encoder fine-tuning configurations under identical
conditions (same data, architecture, epochs, seed) to identify which best
preserves pretrained visual representations while adapting to the bilingual
Pinyin vocabulary.

Run types (--freeze Ã— --lr_schedule):
  1. full_freeze  + fixed
      Encoder weights frozen (requires_grad=False); decoder trained at
      constant LR with no scheduler.  Upper bound on forgetting prevention.

  2. full_freeze  + cosine  (or plateau)
      Encoder frozen; decoder/CTC-head LR decayed by CosineAnnealingLR
      (T_max=10) or ReduceLROnPlateau when --lr_schedule plateau is given.

  3. partial_freeze + fixed
      Encoder trained at base_lr Ã— encoder_lr_scale (default 0.1); decoder
      at base_lr; constant LR throughout.  Balances adaptation and retention.

  4. partial_freeze + cosine  (or plateau)
      Same differential-LR split as run 3, but with CosineAnnealingLR
      (T_max=10) or ReduceLROnPlateau scheduler applied to all groups.

All runs share:
  - Joint CTC + Attention loss (50/50)
  - Pure teacher-forcing (consistent with Obj A baseline)
  - AMP (fp16), gradient clipping (max_norm=5.0)
  - Early stopping on validation loss (patience=5); best model saved on val CER

Validation reports aggregate CER and per-language CER (English / Mandarin / Mixed).

Usage:
  python obj_c_train.py --freeze full_freeze    --lr_schedule fixed   --experiment_log outputs/logs/objC_full_freeze_fixed.json
  python obj_c_train.py --freeze full_freeze    --lr_schedule cosine  --experiment_log outputs/logs/objC_full_freeze_cosine.json
  python obj_c_train.py --freeze partial_freeze --lr_schedule fixed   --experiment_log outputs/logs/objC_partial_freeze_fixed.json
  python obj_c_train.py --freeze partial_freeze --lr_schedule cosine  --experiment_log outputs/logs/objC_partial_freeze_cosine.json
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
    parser = ArgumentParser(description="Objective C: Fine-tuning strategy comparison")

    parser.add_argument('--freeze', type=str, default='partial_freeze',
                        choices=['full_freeze', 'partial_freeze'],
                        help=(
                            "full_freeze    : encoder weights frozen (requires_grad=False)\n"
                            "partial_freeze : encoder trained at lr Ã— encoder_lr_scale  (default)"
                        ))

    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['fixed', 'cosine', 'plateau'],
                        help=(
                            "fixed   : constant learning rate, no scheduler\n"
                            "cosine  : CosineAnnealingLR (T_max=10)  (default)\n"
                            "plateau : ReduceLROnPlateau"
                        ))

    # Paths
    parser.add_argument('--train_video_dir',  type=str, default=r"data/raw/CSLR_Strata/Final_Split\train\pt")
    parser.add_argument('--train_token_dir',  type=str, default=r"data/raw/CSLR_Strata/Final_Split\train\npz")
    parser.add_argument('--val_video_dir',    type=str, default=r"data/raw/CSLR_Strata/Final_Split\val\pt")
    parser.add_argument('--val_token_dir',    type=str, default=r"data/raw/CSLR_Strata/Final_Split\val\npz")
    parser.add_argument('--pretrain_encoder', type=str, default=r"checkpoints/pretrained/LRS2_V_WER26.1/model.pth")
    parser.add_argument('--resume',           type=str, default=None)
    parser.add_argument('--best_model_path',  type=str, default="vsr_c_best.pth")
    parser.add_argument('--experiment_log',   type=str,
                        default=None,
                        help="Path to output JSON log. Defaults to "
                             "outputs/logs/objC_{freeze}_{lr_schedule}.json")

    # Training
    parser.add_argument('--epochs',                  type=int,   default=100)
    parser.add_argument('--batch_size',              type=int,   default=4)
    parser.add_argument('--lr',                      type=float, default=3e-5,
                        help="Base learning rate. Encoder may be scaled depending on strategy.")
    parser.add_argument('--encoder_lr_scale',        type=float, default=0.1,
                        help="Encoder LR multiplier for differential_lr strategy.")
    parser.add_argument('--grad_clip',               type=float, default=5.0)
    parser.add_argument('--seed',                    type=int,   default=42)
    parser.add_argument('--num_workers',             type=int,   default=0)
    parser.add_argument('--amp',                     action='store_true', default=True)
    parser.add_argument('--early_stopping_patience', type=int,   default=5)

    parser.add_argument('--rlrop_factor',   type=float, default=0.5,
                        help="ReduceLROnPlateau: factor by which LR is reduced (default: 0.5).")
    parser.add_argument('--rlrop_patience', type=int,   default=3,
                        help="ReduceLROnPlateau: epochs with no improvement before reducing LR (default: 3).")
    parser.add_argument('--rlrop_min_lr',   type=float, default=1e-7,
                        help="ReduceLROnPlateau: lower bound on LR (default: 1e-7).")

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


def load_pretrained(model: E2E, path: str, device: str):
    print(f"Loading pretrained weights from: {path}")
    sd = torch.load(path, map_location=device)
    md = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in md and md[k].shape == v.shape}
    md.update(matched)
    model.load_state_dict(md, strict=False)
    print(f"  Matched {len(matched)}/{len(sd)} layers")


def build_optimizer(model: E2E, args) -> torch.optim.Optimizer:
    """Build optimizer according to freeze strategy.

    full_freeze    : encoder frozen (no grad), single Adam group for decoder/CTC.
    partial_freeze : two param groups â€” encoder at lr Ã— encoder_lr_scale, rest at lr.
    """
    encoder_params    = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    non_encoder_params = [p for p in model.parameters()
                          if id(p) not in encoder_param_ids]

    if args.freeze == 'full_freeze':
        for p in encoder_params:
            p.requires_grad_(False)
        print(f"  [full_freeze] {sum(p.numel() for p in encoder_params):,} "
              "encoder params frozen")
        optimizer = torch.optim.Adam(non_encoder_params, lr=args.lr)

    else:  # partial_freeze
        enc_lr = args.lr * args.encoder_lr_scale
        optimizer = torch.optim.Adam([
            {'params': encoder_params,     'lr': enc_lr},
            {'params': non_encoder_params, 'lr': args.lr},
        ])
        print(f"  [partial_freeze] encoder lr={enc_lr:.2e}, decoder lr={args.lr:.2e}")

    return optimizer


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
    return ys[0, 1:].cpu().tolist(), found_eos


# â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_one_epoch(model: E2E, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    args, scaler=None, epoch: int = 0) -> tuple:
    model.train()
    # Keep encoder frozen if requested (guard against accidental grad accumulation)
    if args.freeze == 'full_freeze':
        model.encoder.eval()

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
            loss       = 0.3 * ctc_loss + 0.7 * att_loss

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

            loss    = 0.3 * ctc_loss + 0.7 * att_loss
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

    # Auto-derive experiment log path if not explicitly set
    if args.experiment_log is None:
        args.experiment_log = (
            f"outputs/logs/objC_{args.freeze}_{args.lr_schedule}.json"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print(f"Objective C  â€”  freeze: {args.freeze.upper()}  |  lr_schedule: {args.lr_schedule.upper()}")
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
        load_pretrained(model, args.pretrain_encoder, args.device)

    optimizer = build_optimizer(model, args)
    if args.lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.rlrop_factor,
            patience=args.rlrop_patience, min_lr=args.rlrop_min_lr, verbose=True
        )
        print(f"Scheduler: ReduceLROnPlateau  factor={args.rlrop_factor}  "
              f"patience={args.rlrop_patience}  min_lr={args.rlrop_min_lr}")
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        print("Scheduler: CosineAnnealingLR  T_max=10")
    else:  # fixed
        scheduler = None
        print("Scheduler: None (fixed LR)")
    scaler    = torch.cuda.amp.GradScaler() if args.amp else None

    os.makedirs(os.path.dirname(args.experiment_log), exist_ok=True)

    best_cer      = float('inf')
    best_val_loss = float('inf')
    no_improve    = 0
    results_log   = []
    best_epoch    = None

    for epoch in range(args.epochs):
        # Report current LRs for each param group
        lr_info = "  |  ".join(
            f"group{i} lr={g['lr']:.2e}"
            for i, g in enumerate(optimizer.param_groups)
        )
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"[{args.freeze} | {args.lr_schedule}]  {lr_info}")

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
            "epoch":       epoch + 1,
            "freeze":      args.freeze,
            "lr_schedule": args.lr_schedule,
            "train_loss":  tr_loss,
            "train_acc":   tr_acc,
            **val_m,
        })

        # â”€â”€ Save best-CER checkpoint only â”€â”€
        val_cer = val_m['val_cer']
        if val_cer < best_cer:
            best_cer = val_cer
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.best_model_path)
            print(f"âœ… Best CER {best_cer:.4f} â†’ {args.best_model_path}")

        # â”€â”€ Early stopping on val loss (more stable than CER in early epochs) â”€â”€
        val_loss = val_m['val_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1
            print(f"âš ï¸  No val-loss improvement for {no_improve} epoch(s) (best={best_val_loss:.4f})")

        if args.lr_schedule == 'plateau':
            scheduler.step(val_m['val_loss'])
        elif scheduler is not None:
            scheduler.step()

        if no_improve >= args.early_stopping_patience:
            print("ðŸ›‘ Early stopping triggered")
            break

    # Save experiment log with best checkpoint info
    experiment_summary = {
        "freeze": args.freeze,
        "lr_schedule": args.lr_schedule,
        "best_model_path": args.best_model_path,
        "best_cer": best_cer,
        "best_epoch": best_epoch,
        "results_log": results_log
    }
    with open(args.experiment_log, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    print(f"\nResults saved to {args.experiment_log}")


if __name__ == "__main__":
    main()




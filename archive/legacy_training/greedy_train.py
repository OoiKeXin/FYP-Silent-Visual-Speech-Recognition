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

import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from tqdm import tqdm

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss

# ================= CONFIG =================
TRAIN_VIDEO_DIR = r"data/raw/CSLR_Strata/Final_Split\train\pt"
TRAIN_TOKEN_DIR = r"data/raw/CSLR_Strata/Final_Split\train\npz"

VAL_VIDEO_DIR   = r"data/raw/CSLR_Strata/Final_Split\val\pt"
VAL_TOKEN_DIR   = r"data/raw/CSLR_Strata/Final_Split\val\npz"

PRETRAIN_ENCODER = r"checkpoints/pretrained/LRS2_V_WER26.1/model.pth"

PAD_ID = 0
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EARLY_STOPPING_PATIENCE = 3
BEST_MODEL_PATH = "vsr_best.pth"

DEBUG = True
PRINT_SAMPLE_EVERY = 20

torch.backends.cudnn.benchmark = True
# ==========================================

# ================= CHAR LIST =================
CHAR_LIST = [
    "<blank>", "<unk>", "'", "0","1","2","3","4","5","6","7","8","9",
    "<space>",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "<eos>"
]
VOCAB_SIZE = len(CHAR_LIST)
EOS_ID = VOCAB_SIZE - 1
# ==========================================

# ================= DATASET =================
class VSRDataset(Dataset):
    def __init__(self, pt_dir, npz_dir, training=True):
        self.pt_map = {os.path.splitext(f)[0]: os.path.join(pt_dir, f)
                       for f in os.listdir(pt_dir) if f.endswith(".pt")}
        self.npz_map = {os.path.splitext(f)[0]: os.path.join(npz_dir, f)
                        for f in os.listdir(npz_dir) if f.endswith(".npz")}

        self.keys = sorted(set(self.pt_map) & set(self.npz_map))
        assert len(self.keys) > 0, "No matching data found"

        self.training = training

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        video = torch.load(self.pt_map[key]).float()
        tokens = torch.tensor(np.load(self.npz_map[key])["token_ids"]).long()

        if video.ndim == 3:
            video = video.unsqueeze(1)
        if video.shape[0] != 1 and video.shape[1] == 1:
            video = video.permute(1, 0, 2, 3)
        if video.shape[0] != 1:
            video = video.mean(dim=0, keepdim=True)

        return video, tokens


def collate_fn(batch):
    videos, tokens = zip(*batch)

    max_T = max(v.shape[1] for v in videos)
    max_L = max(t.shape[0] for t in tokens)

    padded_videos = torch.zeros(len(videos), 1, max_T, videos[0].shape[2], videos[0].shape[3])
    padded_tokens = torch.full((len(tokens), max_L), PAD_ID, dtype=torch.long)

    for i, (v, t) in enumerate(zip(videos, tokens)):
        padded_videos[i, :, :v.shape[1]] = v
        padded_tokens[i, :t.shape[0]] = t

    return padded_videos, padded_tokens
# ==========================================

# ================= MODEL ==================
def build_model():
    args = Namespace(
        adim=256,
        aheads=4,
        elayers=12,
        eunits=2048,
        dlayers=6,
        dunits=2048,
        dropout_rate=0.1,
        transformer_attn_dropout_rate=0.1,
        transformer_input_layer="conv3d",
        transformer_encoder_attn_layer_type="rel_mha",
        macaron_style=True,
        use_cnn_module=True,
        cnn_module_kernel=31,
        a_upsample_ratio=1,
        relu_type="swish",
        mtlalpha=0.3,
        lsm_weight=0.1,
        transformer_length_normalized_loss=False,
        ctc_type="warpctc",
        report_cer=False,
        report_wer=False,
        char_list=CHAR_LIST,
        sym_blank="<blank>",
        sym_space="<space>",
    )

    model = E2E(VOCAB_SIZE, args).to(DEVICE)

    model.criterion = LabelSmoothingLoss(
        size=VOCAB_SIZE,
        padding_idx=PAD_ID,
        smoothing=args.lsm_weight,
        normalize_length=True,
    )

    return model


def load_pretrained(model):
    print("Loading pretrained weights...")
    sd = torch.load(PRETRAIN_ENCODER, map_location=DEVICE)
    md = model.state_dict()

    matched = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
    md.update(matched)
    model.load_state_dict(md, strict=False)
    print(f"Loaded {len(matched)} layers from pretrained model")
# ==========================================

# ================= TOKEN HELPERS =================
def tokens_to_text(tokens):
    text = []
    for t in tokens:
        if t == PAD_ID:
            continue
        if t == EOS_ID:
            break
        char = CHAR_LIST[t]
        if char == "<space>":
            text.append(" ")
        elif char not in ["<blank>", "<unk>"]:
            text.append(char)
    return "".join(text)


def compute_cer(pred_tokens, ref_tokens):
    from Levenshtein import distance as lev_distance

    pred_text = tokens_to_text(pred_tokens)
    ref_text = tokens_to_text(ref_tokens)

    if len(ref_text) == 0:
        return 0 if len(pred_text) == 0 else 1

    return lev_distance(pred_text, ref_text) / len(ref_text)
# ==========================================

# ================= TRAIN ==================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, total_tokens, total_correct = 0, 0, 0

    for step, (videos, tokens) in enumerate(tqdm(loader)):
        videos = videos.to(DEVICE, non_blocking=True)
        tokens = tokens.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Encoder
        enc_out, enc_mask = model.encoder(videos, None)

        # ================= CTC LOSS =================
        input_lengths = torch.full((videos.size(0),), enc_out.size(1), dtype=torch.long, device=DEVICE)
        target_lengths = (tokens != PAD_ID).sum(dim=1)

        input_lengths = torch.full(
            (videos.size(0),),
            enc_out.size(1),
            dtype=torch.long,
            device=DEVICE
        )

        ctc_loss = model.ctc(enc_out, input_lengths, tokens)

        # ================= ATTENTION LOSS =================
        ys_in = tokens[:, :-1]
        ys_out = tokens[:, 1:]

        dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
        att_loss = model.criterion(dec_out, ys_out)

        # ================= COMBINE =================
        loss = 0.3 * ctc_loss + 0.7 * att_loss

        loss.backward()
        optimizer.step()

        mask = ys_out != PAD_ID
        preds = dec_out.argmax(dim=2)
        correct = (preds == ys_out) & mask

        total_loss += loss.item() * mask.sum().item()
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

        if DEBUG and step % PRINT_SAMPLE_EVERY == 0:
            print("\n[Train Sample Debug]")
            print("GT TEXT:", tokens_to_text(tokens[0].cpu().tolist()))
            print("PR TEXT:", tokens_to_text(preds[0].cpu().tolist()))

    return total_loss / total_tokens, total_correct / total_tokens
# ==========================================

# ================= VALIDATE ==================
def validate(model, loader):
    model.eval()
    total_loss, total_tokens, total_correct = 0, 0, 0
    total_cer = 0
    num_samples = 0

    with torch.no_grad():
        for step, (videos, tokens) in enumerate(loader):
            videos = videos.to(DEVICE, non_blocking=True)
            tokens = tokens.to(DEVICE, non_blocking=True)

            enc_out, enc_mask = model.encoder(videos, None)
            
            ys_in = tokens[:, :-1]
            ys_out = tokens[:, 1:]

            dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
            loss = model.criterion(dec_out, ys_out)

            mask = ys_out != PAD_ID
            preds = dec_out.argmax(dim=2)
            correct = (preds == ys_out) & mask

            total_loss += loss.item() * mask.sum().item()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            for i in range(tokens.size(0)):
                cer = compute_cer(preds[i].cpu().tolist(), tokens[i].cpu().tolist())
                total_cer += cer
                num_samples += 1

            if DEBUG and step % PRINT_SAMPLE_EVERY == 0:
                print("\n[Val Sample Debug]")
                print("GT TEXT:", tokens_to_text(tokens[0].cpu().tolist()))
                print("PR TEXT:", tokens_to_text(preds[0].cpu().tolist()))
                print("CER:", compute_cer(preds[0].cpu().tolist(), tokens[0].cpu().tolist()))

    return total_loss / total_tokens, total_correct / total_tokens, total_cer / num_samples
# ==========================================

# ================= MAIN ==================
def main():
    print("Device:", DEVICE)

    train_set = VSRDataset(TRAIN_VIDEO_DIR, TRAIN_TOKEN_DIR, training=True)
    val_set   = VSRDataset(VAL_VIDEO_DIR, VAL_TOKEN_DIR, training=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = build_model()
    load_pretrained(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer)
        va_loss, va_acc, va_cer = validate(model, val_loader)

        print(f"Train Loss {tr_loss:.4f} | Acc {tr_acc:.4f}")
        print(f"Val   Loss {va_loss:.4f} | Acc {va_acc:.4f} | CER {va_cer:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("âœ… Saved best model")
        else:
            epochs_no_improve += 1
            print(f"âš ï¸ No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("ðŸ›‘ Early stopping triggered")
            break

        torch.save(model.state_dict(), f"vsr_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()




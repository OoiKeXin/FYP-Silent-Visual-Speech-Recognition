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
Beam search training/validation for VSR model.
This script is based on autogressive_train.py but uses beam search decoding for validation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
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
BATCH_SIZE = 4
EPOCHS = 20
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
BLANK_ID = CHAR_LIST.index("<blank>")
# ==========================================

# ================= DATASET =================
class VSRDataset(Dataset):
    def __init__(self, pt_dir, npz_dir):
        self.pt_map = {os.path.splitext(f)[0]: os.path.join(pt_dir, f)
                       for f in os.listdir(pt_dir) if f.endswith(".pt")}
        self.npz_map = {os.path.splitext(f)[0]: os.path.join(npz_dir, f)
                        for f in os.listdir(npz_dir) if f.endswith(".npz")}

        self.keys = sorted(set(self.pt_map) & set(self.npz_map))
        assert len(self.keys) > 0, "No matching data found"

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        video = torch.load(self.pt_map[key]).float()
        tokens = torch.tensor(np.load(self.npz_map[key])["token_ids"]).long()

        # Ensure EOS exists
        if tokens[-1] != EOS_ID:
            tokens = torch.cat([tokens, torch.tensor([EOS_ID])])

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
        lsm_weight=0.0,
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

    for param in model.encoder.parameters():
        param.requires_grad = False
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

# ================= BEAM SEARCH DECODING ==================
def beam_search_decode(model, enc_out, enc_mask, beam_width=5, max_len=100):
    device = enc_out.device
    sequences = [([model.sos], 0.0)]  # (tokens, score)
    completed = []

    greedy_debug_printed = False
    for step in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == EOS_ID:
                completed.append((seq, score))
                continue
            ys = torch.tensor([seq], device=device)
            dec_out, _ = model.decoder(ys, None, enc_out, enc_mask)
            probs = F.log_softmax(dec_out[:, -1, :], dim=-1)
            # Penalize EOS at the first step
            if step == 0:
                probs[0, EOS_ID] -= 100.0
            topk_probs, topk_ids = probs.topk(beam_width, dim=-1)
            if step == 0 and not greedy_debug_printed:
                print("\n================ [BEAM DEBUG] Step 0 top-k tokens ================")
                for k in range(beam_width):
                    print(f"  Token: {topk_ids[0, k].item()} (char: {CHAR_LIST[topk_ids[0, k].item()]}) | LogProb: {topk_probs[0, k].item():.4f} | Prob: {torch.exp(topk_probs[0, k]).item():.4f}")
                greedy_token = torch.argmax(probs, dim=-1).item()
                print(f"[GREEDY DEBUG] Step 0 top-1 token: {greedy_token} (char: {CHAR_LIST[greedy_token]}) | LogProb: {probs[0, greedy_token].item():.4f} | Prob: {torch.exp(probs[0, greedy_token]).item():.4f}")
                greedy_debug_printed = True
            for k in range(beam_width):
                candidate = (seq + [topk_ids[0, k].item()], score + topk_probs[0, k].item())
                all_candidates.append(candidate)
        if not all_candidates:
            break
        # Keep top beam_width sequences
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]
        # Debug: print current step candidates
        if step == 0 or step == max_len - 1:
            print(f"[BEAM DEBUG] Step {step} candidates:")
            for cand_seq, cand_score in sequences:
                print("  Seq:", cand_seq, "Score:", cand_score)
    completed += [seq for seq in sequences if seq[0][-1] == EOS_ID]
    if not completed:
        completed = sequences
    # Remove SOS and return best
    best_seq = max(completed, key=lambda tup: tup[1])[0][1:]
    # Debug: print final best sequence
    print("[BEAM DEBUG] Final best sequence (raw):", best_seq)
    # Fallback: if best_seq is empty, return first token after SOS if available
    if len(best_seq) == 0 and len(completed) > 0 and len(completed[0][0]) > 1:
        print("[BEAM DEBUG] Fallback: returning first token after SOS.")
        return [completed[0][0][1]]
    return best_seq
# ==========================================

# ================= TRAIN ==================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, total_tokens, total_correct = 0, 0, 0

    for videos, tokens in tqdm(loader):
        videos = videos.to(DEVICE)
        tokens = tokens.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        enc_out, enc_mask = model.encoder(videos, None)

        input_lengths = torch.full((videos.size(0),), enc_out.size(1), dtype=torch.long, device=DEVICE)

        ctc_loss = model.ctc(enc_out, input_lengths, tokens)

        ys_in = tokens[:, :-1]
        ys_out = tokens[:, 1:]

        dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
        att_loss = model.criterion(dec_out, ys_out)

        loss = 0.5 * ctc_loss + 0.5 * att_loss

        loss.backward()
        optimizer.step()

        mask = ys_out != PAD_ID
        preds = dec_out.argmax(dim=2)
        correct = (preds == ys_out) & mask

        total_loss += loss.item() * mask.sum().item()
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

    return total_loss / total_tokens, total_correct / total_tokens
# ==========================================

# ================= VALIDATE ==================
def validate(model, loader):
    model.eval()

    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_cer = 0
    num_samples = 0

    with torch.no_grad():
        for videos, tokens in loader:
            videos = videos.to(DEVICE)
            tokens = tokens.to(DEVICE)

            enc_out, enc_mask = model.encoder(videos, None)

            if enc_mask is None:
                enc_mask = torch.ones(enc_out.size(0), enc_out.size(1), device=DEVICE).bool()

            input_lengths = torch.full((videos.size(0),), enc_out.size(1), dtype=torch.long, device=DEVICE)

            ctc_loss = model.ctc(enc_out, input_lengths, tokens)

            ys_in = tokens[:, :-1]
            ys_out = tokens[:, 1:]

            dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
            att_loss = model.criterion(dec_out, ys_out)

            loss = 0.5 * ctc_loss + 0.5 * att_loss

            mask = ys_out != PAD_ID
            preds = dec_out.argmax(dim=2)
            correct = (preds == ys_out) & mask

            total_loss += loss.item() * mask.sum().item()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            # Beam search decoding
            for i in range(videos.size(0)):
                enc_out_i = enc_out[i:i+1]
                enc_mask_i = enc_mask[i:i+1]
                pred_tokens = beam_search_decode(model, enc_out_i, enc_mask_i)
                ref_tokens = tokens[i].cpu().tolist()
                pred_clean = [t for t in pred_tokens if t != PAD_ID and t != model.sos]
                ref_clean = [t for t in ref_tokens if t != PAD_ID and t != model.sos]
                cer = compute_cer(pred_clean, ref_clean)
                total_cer += cer
                num_samples += 1
                if DEBUG and i == 0:
                    print("\n[VAL SAMPLE]")
                    print("GT :", tokens_to_text(ref_clean))
                    print("PR :", tokens_to_text(pred_clean))
                    print("CER:", cer)
    return (
        total_loss / total_tokens,
        total_correct / total_tokens,
        total_cer / num_samples
    )
# ==========================================

# ================= MAIN ==================
def main():
    print("Device:", DEVICE)

    train_set = VSRDataset(TRAIN_VIDEO_DIR, TRAIN_TOKEN_DIR)
    val_set   = VSRDataset(VAL_VIDEO_DIR, VAL_TOKEN_DIR)

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




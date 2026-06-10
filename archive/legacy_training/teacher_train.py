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
EPOCHS = 100
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EARLY_STOPPING_PATIENCE = 7
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
        
         # âœ… ADD EOS HERE
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
    from Levenshtein import editops

    pred_text = tokens_to_text(pred_tokens)
    ref_text = tokens_to_text(ref_tokens)

    if len(ref_text) == 0:
        return 0 if len(pred_text) == 0 else 1

    ops = editops(ref_text, pred_text)
    S = sum(1 for op, _, _ in ops if op == 'replace')
    D = sum(1 for op, _, _ in ops if op == 'delete')
    I = sum(1 for op, _, _ in ops if op == 'insert')
    C = len(ref_text) - S - D  # correct = ref_len - substitutions - deletions

    denom = S + D + I + C
    return (S + D + I) / denom if denom > 0 else 0
# ==========================================

# ================= BEAM SEARCH ==================
def hybrid_beam_search_ctc_att(
    model,
    enc_out,
    enc_mask,
    beam_size=3,          # ðŸ”½ reduce for speed
    max_len=30,           # ðŸ”½ prevent long loops
    ctc_weight=0.3,
    att_weight=0.7,
    length_penalty=0.6
):
    device = enc_out.device
    model.eval()

    # Precompute CTC log probs
    ctc_log_probs = model.ctc.log_softmax(enc_out)[0]  # (T, V)

    beam = [([BLANK_ID], 0.0)]

    for step in range(max_len):

        if DEBUG and step % 5 == 0:
            print(f"[Beam] step={step} | beam size={len(beam)}")
            print("  top seq:", beam[0][0][:10])

        new_beam = []

        for seq, score in beam:

            # âœ… Stop if EOS
            if seq[-1] == EOS_ID:
                new_beam.append((seq, score))
                continue

            ys = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

            dec_out, _ = model.decoder(ys, None, enc_out, enc_mask)
            att_log_probs = F.log_softmax(dec_out[:, -1, :], dim=-1)

            # âœ… Encourage EOS after some length
            if len(seq) > 10:
                att_log_probs[0, EOS_ID] += 2.0

            topk_logp, topk_ids = torch.topk(att_log_probs, beam_size)

            for k in range(beam_size):
                token = topk_ids[0, k].item()
                att_lp = topk_logp[0, k].item()

                # âœ… Prevent repetition loop
                if len(seq) > 2 and token == seq[-1]:
                    continue
                
                if token in seq[-3:]:
                    continue

                ctc_lp = ctc_log_probs[-1, token].item()

                combined = att_weight * att_lp + ctc_weight * ctc_lp
                new_seq = seq + [token]

                norm_score = (score + combined) / (len(new_seq) ** length_penalty)

                new_beam.append((new_seq, norm_score))

        # sort + prune
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]

        # âœ… Debug best sequence
        if DEBUG and step % 5 == 0:
            best_seq = beam[0][0]
            print("  best seq:", best_seq[:20])
            print("  decoded :", tokens_to_text(best_seq))

        # âœ… Early stop if all ended
        if all(seq[-1] == EOS_ID for seq, _ in beam):
            print("[Beam] Early stop: all beams ended")
            break

    print("[Beam] Finished decoding. Final length:", len(beam[0][0]))
    return beam[0][0]
# ==========================================

# ================= TRAIN ==================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss, total_tokens, total_correct = 0, 0, 0

    for step, (videos, tokens) in enumerate(tqdm(loader)):
        videos = videos.to(DEVICE)
        tokens = tokens.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        enc_out, enc_mask = model.encoder(videos, None)

        input_lengths = torch.full((videos.size(0),), enc_out.size(1), dtype=torch.long, device=DEVICE)

        ctc_loss = model.ctc(enc_out, input_lengths, tokens)

        sos_tokens = torch.full((tokens.size(0), 1), model.sos, dtype=tokens.dtype, device=DEVICE)
        ys_in = torch.cat([sos_tokens, tokens[:, :-1]], dim=1)
        ys_out = tokens

        dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
        att_loss = model.criterion(dec_out, ys_out)

        loss = 0.3 * ctc_loss + 0.7 * att_loss

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

    print("===== ðŸ” START VALIDATION =====")

    total_loss = 0
    total_correct = 0
    total_tokens = 0
    total_cer = 0
    num_samples = 0

    with torch.no_grad():
        for step, (videos, tokens) in enumerate(loader):

            if DEBUG:
                print(f"[VAL] Batch {step} start")
                print("  videos:", videos.shape, "tokens:", tokens.shape)

            videos = videos.to(DEVICE)
            tokens = tokens.to(DEVICE)

            enc_out, enc_mask = model.encoder(videos, None)

            if enc_mask is None:
                enc_mask = torch.ones(enc_out.size(0), enc_out.size(1), device=DEVICE).bool()

            input_lengths = torch.full((videos.size(0),), enc_out.size(1), dtype=torch.long, device=DEVICE)

            ctc_loss = model.ctc(enc_out, input_lengths, tokens)

            sos_tokens = torch.full((tokens.size(0), 1), model.sos, dtype=tokens.dtype, device=DEVICE)
            ys_in = torch.cat([sos_tokens, tokens[:, :-1]], dim=1)
            ys_out = tokens

            dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
            att_loss = model.criterion(dec_out, ys_out)

            loss = 0.3 * ctc_loss + 0.7 * att_loss

            mask = ys_out != PAD_ID
            preds = dec_out.argmax(dim=2)
            correct = (preds == ys_out) & mask

            total_loss += loss.item() * mask.sum().item()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            for i in range(videos.size(0)):
                enc_out_i = enc_out[i:i+1]
                enc_mask_i = enc_mask[i:i+1]

                pred_tokens = preds[i].cpu().tolist()

                ref_tokens = tokens[i].cpu().tolist()

                pred_clean = [t for t in pred_tokens if t != PAD_ID]
                ref_clean = [t for t in ref_tokens if t != PAD_ID]

                cer = compute_cer(pred_clean, ref_clean)

                total_cer += cer
                num_samples += 1

                # âœ… ADD THIS DEBUG
                if DEBUG and i == 0:
                    print("\n[VAL SAMPLE]")
                    print("GT :", tokens_to_text(ref_clean))
                    print("PR :", tokens_to_text(pred_clean))
                    print("CER:", cer)

    return total_loss / total_tokens, total_correct / total_tokens, total_cer / num_samples
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer)
        va_loss, va_acc, va_cer = validate(model, val_loader)

        print(f"Train Loss {tr_loss:.4f} | Acc {tr_acc:.4f}")
        print(f"Val   Loss {va_loss:.4f} | Acc {va_acc:.4f} | CER {va_cer:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(va_loss)

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




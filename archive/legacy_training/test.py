
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss

# ================= CONFIG =================
VAL_VIDEO_DIR   = r"CSLR_Strata/Final_Split/test/pt"
VAL_TOKEN_DIR   = r"CSLR_Strata/Final_Split/test/npz"

PAD_ID = 0
BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "vsr_epoch12.pth"

# ================= CHAR LIST =================
CHAR_LIST = [
    "<blank>", "<unk>", "'", "0","1","2","3","4","5","6","7","8","9",
    "<space>",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "<eos>"
]
VOCAB_SIZE = len(CHAR_LIST)

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
        if video.ndim == 3:
            video = video.unsqueeze(1)
        if video.shape[0] != 1 and video.shape[1] == 1:
            video = video.permute(1, 0, 2, 3)
        if video.shape[0] != 1:
            video = video.mean(dim=0, keepdim=True)
        T = video.shape[1]
        tokens = tokens[:T]
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

# ================= MODEL =================
def build_model():
    from argparse import Namespace
    args = Namespace(
        adim=256, aheads=4, elayers=12, eunits=2048,
        dlayers=6, dunits=2048, dropout_rate=0.0,
        transformer_attn_dropout_rate=0.0,
        transformer_input_layer="conv3d",
        transformer_encoder_attn_layer_type="rel_mha",
        macaron_style=True, use_cnn_module=True, cnn_module_kernel=31,
        a_upsample_ratio=1, relu_type="swish", mtlalpha=0.1, lsm_weight=0.0,
        transformer_length_normalized_loss=False, ctc_type="warpctc",
        report_cer=False, report_wer=False,
        char_list=CHAR_LIST, sym_blank="<blank>", sym_space="<space>",
    )
    model = E2E(VOCAB_SIZE, args).to(DEVICE)
    model.criterion = LabelSmoothingLoss(
        size=VOCAB_SIZE, padding_idx=PAD_ID, smoothing=args.lsm_weight, normalize_length=True
    )
    return model

def safe_tokens(tokens):
    tokens = tokens.clone()
    tokens[tokens < 0] = PAD_ID
    return torch.clamp(tokens, 0, VOCAB_SIZE - 1)

# ================= CER (pure python) =================
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def tokens_to_text(tokens_batch):
    texts = []
    for tokens in tokens_batch:
        text = ""
        for t in tokens:
            if t == PAD_ID or t == CHAR_LIST.index("<eos>") or t == CHAR_LIST.index("<blank>"):
                continue
            text += CHAR_LIST[t]
        texts.append(text)
    return texts

# ================= EVALUATION =================
def evaluate(model, loader, print_samples=10):
    model.eval()
    total_correct, total_tokens = 0, 0
    total_errors, total_chars = 0, 0
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (videos, tokens) in enumerate(loader):
            videos = videos.to(DEVICE)
            tokens = tokens.to(DEVICE)
            ys_in = safe_tokens(tokens[:, :-1])
            ys_out = tokens[:, 1:]

            enc_out, enc_mask = model.encoder(videos, None)
            dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)

            mask = ys_out != PAD_ID
            preds = dec_out.argmax(dim=2)
            correct = (preds == ys_out) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            # Convert to text for CER
            pred_texts = tokens_to_text(preds.cpu())
            target_texts = tokens_to_text(ys_out.cpu())

            # Print with filenames
            for i, (p, t) in enumerate(zip(pred_texts, target_texts)):
                # Compute the global sample index in the dataset
                global_idx = batch_idx * loader.batch_size + i
                if global_idx >= len(loader.dataset):
                    continue
                filename = loader.dataset.keys[global_idx]

                total_errors += levenshtein(p, t)
                total_chars += len(t)

                # Print predicted vs ground truth for first few samples
                if sample_count < print_samples:
                    print(f"Sample {sample_count+1}: {filename}")
                    print(f"  Ground truth: {t}")
                    print(f"  Predicted   : {p}")
                    print()
                    sample_count += 1

    token_acc = total_correct / total_tokens
    cer_val = total_errors / max(1, total_chars)
    return token_acc, cer_val

# ================= MAIN =================
def main():
    print("Device:", DEVICE)
    val_set = VSRDataset(VAL_VIDEO_DIR, VAL_TOKEN_DIR)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn)
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    token_acc, cer_val = evaluate(model, val_loader)
    print(f"Token-level accuracy: {token_acc:.4f}")
    print(f"CER: {cer_val:.4f}")

if __name__ == "__main__":
    main()

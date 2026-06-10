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
import platform
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import json

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True   # Ada Lovelace tensor cores
torch.backends.cudnn.allow_tf32 = True
# ================= CONFIG (Argparse) =================
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='checkpoints/pretrained/LRS2_V_WER26.1/model.json')
    parser.add_argument('--train_video_dir', type=str, default=r"data/raw/CSLR_Strata/Final_Split\train\pt")
    parser.add_argument('--train_token_dir', type=str, default=r"data/raw/CSLR_Strata/Final_Split\train\npz")
    parser.add_argument('--val_video_dir', type=str, default=r"data/raw/CSLR_Strata/Final_Split\val\pt")
    parser.add_argument('--val_token_dir', type=str, default=r"data/raw/CSLR_Strata/Final_Split\val\npz")
    parser.add_argument('--pretrain_encoder', type=str, default=r"checkpoints/pretrained/LRS2_V_WER26.1/model.pth")
    parser.add_argument('--resume', type=str, default=None, help='Path to full model checkpoint to resume from (e.g. vsr_best.pth)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers. Keep 0 on Windows to avoid shared memory errors.')
    parser.add_argument('--compile', action='store_true', default=True, help='torch.compile the model (requires PyTorch 2.0+)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--best_model_path', type=str, default="vsr_best.pth")
    parser.add_argument('--print_sample_every', type=int, default=20)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--seed', type=int, default=42)
    # Model hyperparameters
    parser.add_argument('--adim', type=int, default=256)
    parser.add_argument('--aheads', type=int, default=4)
    parser.add_argument('--elayers', type=int, default=12)
    parser.add_argument('--eunits', type=int, default=2048)
    parser.add_argument('--dlayers', type=int, default=6)
    parser.add_argument('--dunits', type=int, default=2048)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--transformer_attn_dropout_rate', type=float, default=0.1)
    parser.add_argument('--lsm_weight', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--transformer_input_layer', type=str, default='conv3d', choices=['conv3d', 'conv2d', 'linear', 'embed'], help='Input subsampling strategy')
    parser.add_argument('--transformer_encoder_attn_layer_type', type=str, default='rel_mha', choices=['mha', 'rel_mha'], help='Attention layer type')
    parser.add_argument('--macaron_style', action='store_true', help='Use macaron style encoder')
    parser.add_argument('--use_cnn_module', action='store_true', help='Use CNN module in encoder')
    parser.add_argument('--cnn_module_kernel', type=int, default=31, help='Kernel size for CNN module')
    parser.add_argument('--relu_type', type=str, default='swish', choices=['relu', 'gelu', 'swish'], help='Activation function')
    parser.add_argument('--normalization', type=str, default='layernorm', choices=['layernorm', 'batchnorm'], help='Normalization layer')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd', 'rmsprop'])
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=[None, 'plateau', 'step', 'cosine', 'onecycle'])
    parser.add_argument('--scheduler_patience', type=int, default=2)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_step_size', type=int, default=5)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--scheduler_max_lr', type=float, default=1e-3, help='Max LR for OneCycleLR')
    parser.add_argument('--scheduler_t_max', type=int, default=10, help='T_max for CosineAnnealingLR')
    parser.add_argument('--beam_width', type=int, default=1)
    parser.add_argument('--ctc_weight', type=float, default=0.5, help='CTC weight for joint decoding')
    parser.add_argument('--length_bonus', type=float, default=0.0, help='Length bonus for decoding')
    parser.add_argument('--length_penalty', type=float, default=0.0)
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy', 'beam', 'lm_beam'])
    parser.add_argument('--rnnlm', type=str, default=None, help='Path to RNNLM model.pth (e.g. data/external/benchmarks/LRS2/language_models/lm_en/model.pth)')
    parser.add_argument('--rnnlm_conf', type=str, default=None, help='Path to RNNLM model.json')
    parser.add_argument('--lm_weight', type=float, default=0.3, help='LM shallow fusion weight (0=disabled, 0.3-0.6 typical)')
    parser.add_argument('--scheduled_sampling', action='store_true', help='Use scheduled sampling during training')
    parser.add_argument('--ss_start_epoch', type=int, default=1, help='Epoch to start scheduled sampling')
    parser.add_argument('--ss_end_epoch', type=int, default=20, help='Epoch to end scheduled sampling (reach prob=1.0)')
    parser.add_argument('--experiment_log', type=str, default=None)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--early_stopping_metric', type=str, default='loss', choices=['loss', 'cer'])
    args = parser.parse_args()
    # Config file support (YAML/JSON)
    if args.config:
        import os
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            import yaml
            with open(args.config, 'r') as f:
                config_args = yaml.safe_load(f)
        else:
            with open(args.config, 'r') as f:
                config_args = json.load(f)
        for k, v in config_args.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args

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
PAD_ID = CHAR_LIST.index("<blank>")
# ==========================================

# ================= SCHEDULED SAMPLING ==================
def get_sampling_prob(epoch, ss_start, ss_end):
    """Calculate probability of using model predictions (vs teacher forcing).
    
    Linear schedule from 0 at ss_start to 1.0 at ss_end.
    """
    if epoch < ss_start:
        return 0.0  # Pure teacher forcing
    if epoch >= ss_end:
        return 1.0  # Pure autoregressive
    return (epoch - ss_start) / (ss_end - ss_start)


def build_scheduled_input(model, enc_out, enc_mask, tokens, sampling_prob, args, device):
    """Build decoder input with scheduled sampling.
    
    Gradually mix ground truth tokens (teacher forcing) with model predictions.
    Returns ys_in (batch, seq_len) for decoder input.
    """
    batch_size, max_len = tokens.shape
    sos_tokens = torch.full((batch_size, 1), model.sos, dtype=tokens.dtype, device=device)
    ys_in = sos_tokens.clone()
    
    # Iterate step by step, deciding whether to use ground truth or model prediction
    for step in range(max_len - 1):
        # Decode current position (no gradients needed for token selection)
        with torch.no_grad():
            dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
        # Get predictions for this step
        preds = dec_out[:, -1, :].argmax(dim=-1)  # (batch,)
        
        # Scheduled sampling: use model prediction with prob `sampling_prob`, else use ground truth
        if sampling_prob > 0:
            use_pred = torch.rand(batch_size, device=device) < sampling_prob  # (batch,) bool
        else:
            use_pred = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Create next token: choose between pred and ground truth
        next_tokens = torch.where(use_pred, preds, tokens[:, step])  # (batch,)
        ys_in = torch.cat([ys_in, next_tokens.unsqueeze(1)], dim=1)
    
    return ys_in
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
            # Optional: print a warning for the first few samples
            if idx < 5:
                print(f"[Warning] EOS appended to sample {key}")

        if video.ndim == 3:
            video = video.unsqueeze(1)
        if video.shape[0] != 1 and video.shape[1] == 1:
            video = video.permute(1, 0, 2, 3)
        if video.shape[0] != 1:
            video = video.mean(dim=0, keepdim=True)

        # Debug: print video shape and tokens for first few samples
        if idx < 3:
            print(f"[DEBUG] Sample {key} video shape: {video.shape}")
            print(f"[DEBUG] Sample {key} tokens: {tokens.tolist()}")
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
def build_model(args):
    model_args = Namespace(
        adim=args.adim,
        aheads=args.aheads,
        elayers=args.elayers,
        eunits=args.eunits,
        dlayers=args.dlayers,
        dunits=args.dunits,
        dropout_rate=args.dropout_rate,
        transformer_attn_dropout_rate=args.transformer_attn_dropout_rate,
        transformer_input_layer=args.transformer_input_layer,
        transformer_encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
        macaron_style=args.macaron_style,
        use_cnn_module=args.use_cnn_module,
        cnn_module_kernel=args.cnn_module_kernel,
        a_upsample_ratio=1,
        relu_type=args.relu_type,
        normalization=args.normalization,
        mtlalpha=0.3,
        lsm_weight=args.lsm_weight,
        transformer_length_normalized_loss=False,
        ctc_type="warpctc",
        report_cer=False,
        report_wer=False,
        char_list=CHAR_LIST,
        sym_blank="<blank>",
        sym_space="<space>",
    )
    model = E2E(VOCAB_SIZE, model_args).to(args.device)
    model.criterion = LabelSmoothingLoss(
        size=VOCAB_SIZE,
        padding_idx=PAD_ID,
        smoothing=args.label_smoothing,
        normalize_length=True,
    )
    return model


def load_pretrained(model, args):
    print("Loading pretrained weights...")
    sd = torch.load(args.pretrain_encoder, map_location=args.device)
    md = model.state_dict()
    matched = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
    md.update(matched)
    model.load_state_dict(md, strict=False)
    print(f"Loaded {len(matched)} layers from pretrained model")
    # Encoder unfrozen â€” use lower LR for encoder via param groups
    # (set up in main())
# ==========================================

# ================= TOKEN HELPERS =================
def tokens_to_text(tokens, skip_sos=False):
    text = []
    for i, t in enumerate(tokens):
        if skip_sos and i == 0 and t == EOS_ID:
            continue  # Skip SOS at position 0 (SOS == EOS in espnet)
        if t == PAD_ID:
            continue
        if t == EOS_ID:
            break
        if t < 0 or t >= len(CHAR_LIST):
            text.append("?")
            continue
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

# ================= GREEDY DECODING ==================
def greedy_decode(model, enc_out, enc_mask, args=None, max_len=None):
    device = enc_out.device
    # Cap max_len to 2x the encoder output length (video frames â†’ subsampled)
    # This prevents runaway generation when EOS is not learned yet
    enc_len = enc_out.size(1)
    if max_len is None:
        max_len = max(10, min(enc_len * 2, 50))
    ys = torch.tensor([[model.sos]], device=device)
    ngram_size = 3
    found_eos = False
    for step in range(max_len):
        dec_out, _ = model.decoder(ys, None, enc_out, enc_mask)
        # Attention log-probs only â€” no CTC mixing during greedy decode.
        # Frame-averaged CTC log-probs are dominated by blank and corrupt the signal.
        # CTC is used only during training loss, not decoding.
        probs = torch.log_softmax(dec_out[:, -1, :], dim=-1)
        # Length bonus
        if args is not None and args.length_bonus != 0:
            probs += args.length_bonus
        # Repetition penalty: penalize only the immediately previous token
        # to prevent "AAAA..." streaks while allowing valid repeated chars
        if ys.size(1) > 1:
            last_token = ys[0, -1].item()
            probs[0, last_token] -= 1.5
        # N-gram blocking: block exact trigrams that already appeared
        if ys.size(1) > ngram_size:
            tokens_list = ys[0].tolist()
            seen_ngrams = set()
            for i in range(1, len(tokens_list) - ngram_size + 1):
                seen_ngrams.add(tuple(tokens_list[i:i + ngram_size]))
            prefix = tuple(tokens_list[-(ngram_size - 1):])
            for cand in range(probs.size(1)):
                if prefix + (cand,) in seen_ngrams:
                    probs[0, cand] -= 100.0
        next_token = probs.argmax(dim=-1)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == EOS_ID:
            found_eos = True
            break
    if not found_eos:
        print(f"[WARNING][decode] EOS not found within max_len={max_len} (enc_len={enc_len}). Output truncated.")
    return ys[0, 1:].cpu().tolist()
# ==========================================

# ================= LM BEAM DECODE ==================
def lm_beam_decode(beam_searcher, enc_out, token_list):
    """Decode a single sample using espnet BatchBeamSearch + RNNLM shallow fusion."""
    from espnet.asr.asr_utils import add_results_to_json
    enc_out_squeezed = enc_out.squeeze(0)  # (T, D)
    nbest_hyps = beam_searcher(enc_out_squeezed)
    nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]
    text = add_results_to_json(nbest_hyps, token_list)
    # Convert text back to token ids using CHAR_LIST
    tokens = []
    i = 0
    text = text.replace('<eos>', '')
    while i < len(text):
        matched = False
        for tok_id, tok in enumerate(token_list):
            if tok in ('<blank>', '<unk>', '<eos>') or len(tok) == 0:
                continue
            if tok == '<space>':
                if text[i] == ' ':
                    tokens.append(tok_id)
                    i += 1
                    matched = True
                    break
            elif text[i:i+len(tok)] == tok:
                tokens.append(tok_id)
                i += len(tok)
                matched = True
                break
        if not matched:
            i += 1
    return tokens
# ==========================================

# ================= TRAIN ==================
def train_one_epoch(model, loader, optimizer, args, scaler=None, epoch=0):
    model.train()
    total_loss, total_tokens, total_correct = 0, 0, 0

    for batch_idx, (videos, tokens) in enumerate(tqdm(loader)):
        videos = videos.to(args.device)
        tokens = tokens.to(args.device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                enc_out, enc_mask = model.encoder(videos, None)
                input_lengths = torch.full((videos.size(0),), enc_out.size(1), dtype=torch.long, device=args.device)
                ctc_loss = model.ctc(enc_out, input_lengths, tokens)
                if args.scheduled_sampling:
                    sampling_prob = get_sampling_prob(epoch, args.ss_start_epoch, args.ss_end_epoch)
                    ys_in = build_scheduled_input(model, enc_out, enc_mask, tokens, sampling_prob, args, args.device)
                    ys_out = tokens
                    if batch_idx == 0:
                        print(f"[Epoch {epoch}] Sampling prob (AMP): {sampling_prob:.3f}")
                else:
                    sos_tokens = torch.full((tokens.size(0), 1), model.sos, dtype=tokens.dtype, device=tokens.device)
                    ys_in = torch.cat([sos_tokens, tokens[:, :-1]], dim=1)
                    ys_out = tokens
                dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
                att_loss = model.criterion(dec_out, ys_out)
                loss = 0.5 * ctc_loss + 0.5 * att_loss
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            enc_out, enc_mask = model.encoder(videos, None)
            input_lengths = torch.full((videos.size(0),), enc_out.size(1), dtype=torch.long, device=args.device)
            ctc_loss = model.ctc(enc_out, input_lengths, tokens)
            
            # Scheduled sampling: gradually transition from teacher forcing to autoregressive
            if args.scheduled_sampling:
                sampling_prob = get_sampling_prob(epoch, args.ss_start_epoch, args.ss_end_epoch)
                ys_in = build_scheduled_input(model, enc_out, enc_mask, tokens, sampling_prob, args, args.device)
                ys_out = tokens
                print(f"[Epoch {epoch}] Sampling prob: {sampling_prob:.3f}")
            else:
                # Pure teacher forcing (original)
                sos_tokens = torch.full((tokens.size(0), 1), model.sos, dtype=tokens.dtype, device=args.device)
                ys_in = torch.cat([sos_tokens, tokens[:, :-1]], dim=1)
                ys_out = tokens
            
            dec_out, _ = model.decoder(ys_in, None, enc_out, enc_mask)
            att_loss = model.criterion(dec_out, ys_out)
            loss = 0.5 * ctc_loss + 0.5 * att_loss
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
def validate(model, loader, args, beam_searcher=None):
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_cer = 0
    num_samples = 0
    samples_to_print = 1000  # Print up to 1000 validation samples
    printed = 0
    with torch.no_grad():
        for batch_idx, (videos, tokens) in enumerate(loader):
            videos = videos.to(args.device)
            tokens = tokens.to(args.device)
            enc_out, enc_mask = model.encoder(videos, None)
            if enc_mask is None:
                enc_mask = torch.ones(enc_out.size(0), 1, enc_out.size(1), device=args.device).bool()
            input_lengths = torch.full((videos.size(0),), enc_out.size(1), dtype=torch.long, device=args.device)
            ctc_loss = model.ctc(enc_out, input_lengths, tokens)
            sos_tokens = torch.full((tokens.size(0), 1), model.sos, dtype=tokens.dtype, device=tokens.device)
            ys_in = torch.cat([sos_tokens, tokens[:, :-1]], dim=1)
            ys_out = tokens
            dec_out, dec_att = model.decoder(ys_in, None, enc_out, enc_mask)
            att_loss = model.criterion(dec_out, ys_out)
            loss = 0.5 * ctc_loss + 0.5 * att_loss
            mask = ys_out != PAD_ID
            preds = dec_out.argmax(dim=2)
            correct = (preds == ys_out) & mask
            total_loss += loss.item() * mask.sum().item()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            # Decoding
            for i in range(videos.size(0)):
                enc_out_i = enc_out[i:i+1]
                enc_mask_i = enc_mask[i:i+1]
                if args.decode_strategy == 'lm_beam' and beam_searcher is not None:
                    pred_tokens = lm_beam_decode(beam_searcher, enc_out_i, CHAR_LIST)
                elif args.decode_strategy == 'beam':
                    pred_tokens = beam_decode(model, enc_out_i, enc_mask_i, args)
                else:
                    pred_tokens = greedy_decode(model, enc_out_i, enc_mask_i, args)
                ref_tokens = tokens[i].cpu().tolist()
                pred_clean = [t for t in pred_tokens if t != PAD_ID and t != model.sos]
                ref_clean = [t for t in ref_tokens if t != PAD_ID and t != model.sos]
                cer = compute_cer(pred_clean, ref_clean)
                total_cer += cer
                num_samples += 1
                # Print all validation predictions vs ground truth
                pred_text = tokens_to_text(pred_clean)
                ref_text = tokens_to_text(ref_clean)
                print(f"[VAL] #{num_samples}\n  REF:  {ref_text}\n  PRED: {pred_text}\n  CER:  {cer:.3f}\n")
    return (
        total_loss / total_tokens,
        total_correct / total_tokens,
        total_cer / num_samples
    )
# ================= BEAM DECODE ==================
def beam_decode(model, enc_out, enc_mask, args):
    # Simple beam search for demonstration; can be improved
    device = enc_out.device
    beam_width = args.beam_width
    max_len = 100
    sequences = [[torch.tensor([model.sos], device=device), 0.0]]
    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1].item() == EOS_ID:
                all_candidates.append((seq, score))
                continue
            dec_out, _ = model.decoder(seq.unsqueeze(0), None, enc_out, enc_mask)
            probs = torch.log_softmax(dec_out[:, -1, :], dim=-1)
            # Optionally combine with CTC score if args provided
            if hasattr(model, 'ctc') and args.ctc_weight > 0:
                ctc_probs = model.ctc.log_softmax(enc_out)[0].mean(dim=0)
                probs = (1 - args.ctc_weight) * probs + args.ctc_weight * ctc_probs
            # Length bonus
            if args.length_bonus != 0:
                probs += args.length_bonus
            topk = torch.topk(probs, beam_width)
            for i in range(beam_width):
                token = topk.indices[0, i]
                candidate = torch.cat([seq, token.unsqueeze(0)])
                candidate_score = score + topk.values[0, i].item()
                all_candidates.append((candidate, candidate_score))
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]
        if all(seq[-1].item() == EOS_ID for seq, _ in sequences):
            break
    best_seq = sequences[0][0]
    return best_seq[1:].cpu().tolist()  # remove SOS
# ==========================================

# ================= MAIN ==================
def main():
    args = get_args()
    # Set seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Device:", args.device)

    # Logging
    log_f = open(args.log_file, 'w') if args.log_file else None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter()
    else:
        tb_writer = None

    train_set = VSRDataset(args.train_video_dir, args.train_token_dir)
    val_set   = VSRDataset(args.val_video_dir, args.val_token_dir)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=args.num_workers, pin_memory=(args.num_workers > 0),
                              persistent_workers=(args.num_workers > 0))
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                              num_workers=args.num_workers, pin_memory=(args.num_workers > 0),
                              persistent_workers=(args.num_workers > 0))

    model = build_model(args)
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        sd = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(sd, strict=False)
        print("Loaded full model checkpoint")
    else:
        load_pretrained(model, args)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile() enabled")
        except Exception as e:
            print(f"torch.compile() failed, running in eager mode: {e}")

    # Differential LR: encoder gets 10x lower LR for fine-tuning
    # Must be built BEFORE torch.compile() to avoid duplicate parameter groups
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    param_groups = [
        {'params': encoder_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr},
    ]

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups)
    else:
        optimizer = torch.optim.Adam(param_groups)

    # Build LM beam searcher if requested
    beam_searcher = None
    if args.decode_strategy == 'lm_beam':
        if args.rnnlm and args.rnnlm_conf:
            from pipelines.model import get_beam_search_decoder
            print(f"Loading RNNLM from {args.rnnlm} ...")
            beam_searcher = get_beam_search_decoder(
                model,
                CHAR_LIST,
                rnnlm=args.rnnlm,
                rnnlm_conf=args.rnnlm_conf,
                penalty=args.length_penalty,
                ctc_weight=args.ctc_weight,
                lm_weight=args.lm_weight,
                beam_size=args.beam_width,
            )
            beam_searcher.to(args.device).eval()
            print(f"LM beam searcher ready (lm_weight={args.lm_weight}, beam={args.beam_width})")
        else:
            print("[WARNING] --decode_strategy lm_beam requested but --rnnlm/--rnnlm_conf not provided. Falling back to greedy.")
            args.decode_strategy = 'greedy'

    # Scheduler
    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max)
    elif args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.scheduler_max_lr, steps_per_epoch=len(train_loader), epochs=args.epochs)

    # AMP
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    best_metric = float('inf')
    epochs_no_improve = 0
    results_log = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, args, scaler, epoch=epoch)
        va_loss, va_acc, va_cer = validate(model, val_loader, args, beam_searcher=beam_searcher)
        print(f"Train Loss {tr_loss:.4f} | Acc {tr_acc:.4f}")
        print(f"Val   Loss {va_loss:.4f} | Acc {va_acc:.4f} | CER {va_cer:.4f}")
        if log_f:
            log_f.write(f"Epoch {epoch+1}: Train Loss {tr_loss:.4f} | Acc {tr_acc:.4f} | Val Loss {va_loss:.4f} | Acc {va_acc:.4f} | CER {va_cer:.4f}\n")
        if tb_writer:
            tb_writer.add_scalar('Loss/train', tr_loss, epoch+1)
            tb_writer.add_scalar('Loss/val', va_loss, epoch+1)
            tb_writer.add_scalar('CER/val', va_cer, epoch+1)
        results_log.append({
            'epoch': epoch+1,
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': va_loss,
            'val_acc': va_acc,
            'val_cer': va_cer
        })
        # Early stopping metric
        metric = va_loss if args.early_stopping_metric == 'loss' else va_cer
        if metric < best_metric:
            best_metric = metric
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.best_model_path)
            print("âœ… Saved best model")
        else:
            epochs_no_improve += 1
            print(f"âš ï¸ No improvement for {epochs_no_improve} epoch(s)")
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(metric)
            else:
                scheduler.step()
        if epochs_no_improve >= args.early_stopping_patience:
            print("ðŸ›‘ Early stopping triggered")
            break
        torch.save(model.state_dict(), f"vsr_epoch{epoch+1}.pth")
    # Save experiment log if requested
    if args.experiment_log:
        with open(args.experiment_log, 'w') as f:
            json.dump(results_log, f, indent=2)
    if log_f:
        log_f.close()
    if tb_writer:
        tb_writer.close()

if __name__ == "__main__":
    main()




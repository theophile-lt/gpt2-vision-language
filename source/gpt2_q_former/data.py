from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext
import csv, time, glob
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoCaptions
import random
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table
from pycocoevalcap.cider.cider import Cider
import json
import tiktoken
from model_BLIP import pool_clip_197_to_33_avg_with_cls


class CocoClipCaptionDataset(Dataset):
    def __init__(self, feats_path, coco_root, ann_path, tokenizer, max_len):
        super().__init__()
        # CLIP features: [N, 768]
        self.feats = torch.load(feats_path, map_location="cpu")
        self.ds = CocoCaptions(root=coco_root, annFile=ann_path)
        assert len(self.ds) == self.feats.size(0), "COCO and CLIP feats length mismatch"
        self.enc = tokenizer
        self.max_len = max_len
        # tiktoken gpt2 end-of-text
        self.eot = self.enc.eot_token

    def __len__(self):
        return self.feats.size(0)

    def _encode_caption(self, text):
        ids = self.enc.encode(text)
        if len(ids) == 0:
            ids = [self.eot]
        ids = ids[: self.max_len - 1] + [self.eot]
        L = len(ids)  
        if L < self.max_len:
            ids = ids + [self.eot] * (self.max_len - L)
        ids = torch.tensor(ids, dtype=torch.long)  
        x = ids[:-1]  # (T-1)
        y = ids[1:]   # (T-1)
        valid_len = max(L - 1, 1)   
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:valid_len] = True
        return x, y, mask

    def __getitem__(self, idx):
        img, caps = self.ds[idx]          # caps: list[str]
        text = random.choice(caps)        # random caption for this image
        x, y, m = self._encode_caption(text)  # x,y: (T-1,), m: (T-1,)
        z = self.feats[idx]                   # (768,)
        z = z.unsqueeze(0)                    # (1, 768) → S=1
        return x, y, m, z

class CocoClipFullTokensDataset(Dataset):
    """
    Utilise les shards générés par precompute_clip_full_tokens.py :
    - tokens_dir/
        - tokens_000000_000064.pt  # [B, L, 768]
        - tokens_000064_000128.pt
        - ...
        - index.json               # liste de dicts avec shard/row/...
    """
    def __init__(self, tokens_dir, coco_root, ann_path, tokenizer, max_len):
        super().__init__()
        self.tokens_dir = tokens_dir
        self.ds = CocoCaptions(root=coco_root, annFile=ann_path)
        self.enc = tokenizer
        self.max_len = max_len
        self.eot = self.enc.eot_token

        index_path = os.path.join(tokens_dir, "index.json")
        with open(index_path, "r") as f:
            self.index = json.load(f)
        assert len(self.index) == len(self.ds), "index.json length mismatch with COCO"

        # cache du shard courant
        self._current_shard_name = None
        self._current_shard_tensor = None

    def __len__(self):
        return len(self.ds)

    def _encode_caption(self, text):
        ids = self.enc.encode(text)
        if len(ids) == 0:
            ids = [self.eot]
        ids = ids[: self.max_len - 1] + [self.eot]
        L = len(ids)
        if L < self.max_len:
            ids = ids + [self.eot] * (self.max_len - L)
        ids = torch.tensor(ids, dtype=torch.long)
        x = ids[:-1]
        y = ids[1:]
        valid_len = max(L - 1, 1)
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:valid_len] = True
        return x, y, mask

    def __getitem__(self, idx):
        img, caps = self.ds[idx]
        text = random.choice(caps)
        x, y, m = self._encode_caption(text)

        entry = self.index[idx]
        shard_name = entry["shard"]
        row = entry["row"]

        if shard_name != self._current_shard_name:
            shard_path = os.path.join(self.tokens_dir, shard_name)
            # shard : (B_shard, 197, 768) = [CLS] + 196 patches
            self._current_shard_tensor = torch.load(shard_path, map_location="cpu")
            self._current_shard_name = shard_name

        full_tokens = self._current_shard_tensor[row].unsqueeze(0)   # (1, L, 768)
        z = pool_clip_197_to_33_avg_with_cls(full_tokens)[0]        # (33, 768) on enlève la batch-dim

        return x, y, m, z



def evaluate_cider(
    model,
    device,
    enc,
    COCO_ROOT,
    CLIP_FULL_DIR,
    max_samples=500,
    max_new_tokens=24,
):
    """
    Évalue CIDEr sur un sous-ensemble du val set COCO en utilisant
    les FULL TOKENS CLIP (CLS + patches) précomputés dans CLIP_FULL_DIR/val.
    - model : raw_model (NON DDP)
    - device : "cuda" ou "cpu"
    - enc : tokenizer tiktoken gpt2
    - COCO_ROOT : racine COCO
    - CLIP_FULL_DIR : racine des tokens CLIP complets
    """
    model.eval()
    cider_scorer = Cider()

    model_dtype = next(model.parameters()).dtype

    # --- COCO val : captions de référence ---
    val_coco = CocoCaptions(
        root=os.path.join(COCO_ROOT, "val2017"),
        annFile=os.path.join(COCO_ROOT, "annotations", "captions_val2017.json"),
    )

    # --- FULL TOKENS CLIP (comme CocoClipFullTokensDataset) ---
    tokens_dir = os.path.join(CLIP_FULL_DIR, "val")
    index_path = os.path.join(tokens_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    assert len(index) == len(val_coco), "index.json length mismatch with COCO val"

    current_shard_name = None
    current_shard_tensor = None  # (B_shard, L, 768)

    gts = {}
    res = {}

    n_eval = min(max_samples, len(val_coco))
    for idx in range(n_eval):
        _, caps = val_coco[idx]
        gts[idx] = caps

        entry = index[idx]
        shard_name = entry["shard"]
        row = entry["row"]

        if shard_name != current_shard_name:
            shard_path = os.path.join(tokens_dir, shard_name)
            current_shard_tensor = torch.load(shard_path, map_location="cpu")
            current_shard_name = shard_name

        full_tokens = current_shard_tensor[row].unsqueeze(0)              # (L, 768)
        z = pool_clip_197_to_33_avg_with_cls(full_tokens).to(
            device=device,
            dtype=model_dtype,
        )

        prompt = "A photo of"
        prompt_ids = enc.encode(prompt)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]  # (1, T_txt)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                # BLIP GPT: patch_tokens first, then input_ids
                logits, _ = model(z, x)
                logits_last = logits[:, -1, :] / 0.8
                probs = F.softmax(logits_last, dim=-1)

                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumprobs = sorted_probs.cumsum(dim=-1)
                cutoff = cumprobs > 0.9
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_idx = torch.multinomial(sorted_probs, 1)
                next_token = sorted_idx.gather(-1, next_idx)

            x = torch.cat([x, next_token], dim=1)


        gen_ids = x[0].tolist()
        caption_ids = gen_ids[len(prompt_ids):]
        caption = enc.decode(caption_ids)
        res[idx] = [caption]

    score, _ = cider_scorer.compute_score(gts, res)
    return score



#-------------------------------------------------------
import time
import tiktoken
import numpy as np

def load_tokens(filename) :
    npt = np.load(filename).astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite :
    def __init__(self, B, T, process_rank, num_processes, split) :
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        #at init load tokens from disk and store them in memory

        #get the shard filenames
        data_root = os.environ.get("FW_OUT_DIR", "edu_fineweb10B")
        shards = os.listdir(data_root)
        shards = [ s for s in shards if split in s]
        shards=sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards=shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process :
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self) :
        #state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])    
        self.current_position =self.B*self.T*self.process_rank

    def next_batch(self) :
        B,T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+ B*T +1]
        x= (buf[:-1]).view(B,T)
        y= (buf[1:]).view(B,T)
        #advance the position in the tensor
        self.current_position+= B*T *self.num_processes
        #if loading the next batch would be out of bounds, reset
        if self.current_position + (B*T * self.num_processes+1) > len(self.tokens) :
            self.current_shard=(self.current_shard+1)%len(self.shards)
            self.tokens=load_tokens(self.shards[self.current_shard])
            self.current_position =self.B*self.T*self.process_rank
        return x,y 

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

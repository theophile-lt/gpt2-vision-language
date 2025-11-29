import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions
import random
from pycocoevalcap.cider.cider import Cider
import json
from model import pool_clip_197_to_33_avg_with_cls


# Loading precomputed CLIP embeddings for COCO

class CocoClipFullTokensDataset(Dataset):

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
            self._current_shard_tensor = torch.load(shard_path, map_location="cpu") 
            self._current_shard_name = shard_name
        z = self._current_shard_tensor[row]                                 
        return x, y, m, z


# Evaluation Metric

def evaluate_cider(
    model,
    device,
    enc,
    COCO_ROOT,
    CLIP_FULL_DIR,
    max_samples=500,
    max_new_tokens=24,
):
    model.eval()
    cider_scorer = Cider()
    model_dtype = next(model.parameters()).dtype
    val_coco = CocoCaptions(
        root=os.path.join(COCO_ROOT, "val2017"),
        annFile=os.path.join(COCO_ROOT, "annotations", "captions_val2017.json"),
    )
    tokens_dir = os.path.join(CLIP_FULL_DIR, "val")
    index_path = os.path.join(tokens_dir, "index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    assert len(index) == len(val_coco), "index.json length mismatch with COCO val"

    current_shard_name = None
    current_shard_tensor = None  
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
        z = current_shard_tensor[row]                      
        z = z.unsqueeze(0).to(device=device, dtype=model_dtype)  
        z = pool_clip_197_to_33_avg_with_cls(z) 
        prompt = "A photo of"
        prompt_ids = enc.encode(prompt)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
        for _ in range(max_new_tokens):
            with torch.no_grad():
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

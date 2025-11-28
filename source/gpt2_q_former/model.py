import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
import inspect
import torch.nn.functional as F 

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT_previous(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)      
        tok_emb = self.transformer.wte(idx)     
        x = tok_emb + pos_emb                   
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

def load_nanogpt_from_ckpt(ckpt_path: str, map_location="cpu") -> GPT_previous:
    # This is your own NanoGPT checkpoint; we allow full unpickling
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    if isinstance(raw, dict) and "model" in raw and "config" in raw:
        sd = raw["model"]
        cfg_raw = raw["config"]
        if isinstance(cfg_raw, GPTConfig):
            cfg = cfg_raw
        elif isinstance(cfg_raw, dict):
            cfg = GPTConfig(**cfg_raw)
        else:
            cfg = GPTConfig(vocab_size=50304)
        print(f"[NanoGPT] Loaded checkpoint {ckpt_path} "
              f"(step={raw.get('step','?')}, val_loss={raw.get('val_loss','?')})")
    else:
        sd = raw
        cfg = GPTConfig(vocab_size=50304)
        print(f"[NanoGPT] Loaded flat state_dict from {ckpt_path}")

    gpt = GPT_previous(cfg)
    # <<< HERE is the key change
    gpt.load_state_dict(sd, strict=False)
    return gpt

class QFormerLayer(nn.Module):
    """
    Un bloc de Q-Former :
      - self-attention sur les queries
      - cross-attention queries -> vision features
      - MLP
    """
    def __init__(self, d, n_heads, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=drop, batch_first=True)

        self.ln2_q = nn.LayerNorm(d)
        self.ln2_v = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=drop, batch_first=True)

        self.ln3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, q, v):
        q2 = self.ln1(q)
        sa_out, _ = self.self_attn(q2, q2, q2)
        q = q + self.drop(sa_out)

        q2 = self.ln2_q(q)
        v2 = self.ln2_v(v)
        ca_out, _ = self.cross_attn(q2, v2, v2)
        q = q + self.drop(ca_out)

        q2 = self.ln3(q)
        q = q + self.drop(self.mlp(q2))
        return q

class BLIP2Bridge(nn.Module):
    """
    Bridge BLIP-2 :
      - projette enc_dim -> d_lm
      - M queries apprenables
      - L couches de Q-Former
    Sortie : (B, M, d_lm) = M pseudo-tokens image dans l'espace du LM.
    """
    def __init__(self, enc_dim, d_lm, n_heads, n_queries=8, n_layers=2, drop=0.1):
        super().__init__()
        self.vis_proj = nn.Linear(enc_dim, d_lm)
        self.n_queries = n_queries
        self.query_tokens = nn.Parameter(torch.randn(n_queries, d_lm))

        self.layers = nn.ModuleList(
            [QFormerLayer(d_lm, n_heads, drop=drop) for _ in range(n_layers)]
        )

    def forward(self, patch_tokens):
        x = self.vis_proj(patch_tokens)  
        B, N, D = x.shape

        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            q = layer(q, x)  

        return q  

class GPT_Caption(nn.Module):
    """
    - patch_tokens : (B, N_img, enc_dim) venant de CLIP (gelé)
    - Bridge BLIP-2 (Q-Former light) -> M pseudo-tokens image en dim n_embd
    - NanoGPT (gelé) prend [image_prefix, texte] via embeddings + wpe
    """

    def __init__(
        self,
        enc_dim: int,
        tokenizer,          
        custom_ckpt: str,
        m_vis_tokens: int = 8,
        use_cls_only: bool = False,
        freeze_lm: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_cls_only = use_cls_only

        self.gpt = load_nanogpt_from_ckpt(custom_ckpt)
        cfg = self.gpt.config
        self.d = cfg.n_embd
        self.block_size = cfg.block_size

        self.bridge = BLIP2Bridge(
            enc_dim=enc_dim,
            d_lm=self.d,
            n_heads=cfg.n_head,
            n_queries=m_vis_tokens,
            n_layers=2,
            drop=0.1,
        )

        self.wte = self.gpt.transformer.wte
        self.wpe = self.gpt.transformer.wpe

        if freeze_lm:
            for p in self.gpt.parameters():
                p.requires_grad_(False)
        for p in self.bridge.parameters():
            p.requires_grad_(True)

    def _decode_transformer(self, full_embeds):
        """Passe les embeddings (B, L, d) dans les blocs GPT."""
        x = full_embeds
        for block in self.gpt.transformer.h:
            x = block(x)
        x = self.gpt.transformer.ln_f(x)
        logits = self.gpt.lm_head(x)
        return logits

    def forward(self, patch_tokens, input_ids, labels=None):
        """
        patch_tokens : (B, N_img, enc_dim)
        input_ids    : (B, T_txt)
        labels       : (B, T_txt) (-100 sur pads)
        """
        B, T_txt = input_ids.shape
        device = input_ids.device

        if patch_tokens.dim() == 2:
            patch_tokens = patch_tokens.unsqueeze(1)
        B_img, N_raw, D_enc = patch_tokens.shape
        assert B_img == B, "batch size image != batch size texte"

        x_img = patch_tokens
        if self.use_cls_only:
            x_img = x_img[:, 0:1, :]

        img_embeds = self.bridge(x_img)
        M = img_embeds.size(1)

        txt_embeds = self.wte(input_ids)

        full_len = M + T_txt
        if full_len > self.block_size:
            cut_txt = self.block_size - M
            txt_embeds = txt_embeds[:, :cut_txt, :]
            input_ids = input_ids[:, :cut_txt]
            if labels is not None:
                labels = labels[:, :cut_txt]
            T_txt = cut_txt
            full_len = M + T_txt

        # --- minimal positional fix ---
        # 1) keep text at positions 0..T_txt-1, as in pretraining
        pos_txt = torch.arange(T_txt, device=device)
        pos_txt_embeds = self.wpe(pos_txt).unsqueeze(0)      # (1, T_txt, d)

        # 2) add position ONLY to text tokens
        txt_embeds = txt_embeds + pos_txt_embeds             # (B, T_txt, d)

        # 3) concatenate image prefix (no pos emb) + text
        full_embeds = torch.cat([img_embeds, txt_embeds], dim=1)  # (B, M+T_txt, d)

        logits = self._decode_transformer(full_embeds)


        loss = None
        if labels is not None:
            # logits_text : (B, T_txt, V), aligné avec labels : (B, T_txt)
            logits_text = logits[:, M:M+T_txt, :]  # on ignore simplement les logits image

            loss = F.cross_entropy(
                logits_text.reshape(-1, logits_text.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )


        return logits, loss

    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW:{use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer


def pool_clip_197_to_33_avg_with_cls(tokens_197: torch.Tensor) -> torch.Tensor:
    """
    tokens_197 : (B, L, D) = [CLS] + N patches, avec N = h*w (grille carrée)
    return     : (B, 33, D) = [CLS] + 32 régions poolées (4x8)
    """
    B, L, D = tokens_197.shape

    # 1) séparer CLS et patches
    cls = tokens_197[:, :1, :]            # (B, 1, D)
    patches = tokens_197[:, 1:, :]        # (B, N, D)
    N = patches.size(1)

    # 2) retrouver la taille de la grille (supposée carrée)
    side = int(round(N ** 0.5))
    assert side * side == N, f"Expected square grid, got N={N}"

    # 3) remettre en (B, D, H, W)
    patches = patches.view(B, side, side, D)   # (B, H, W, D)
    patches = patches.permute(0, 3, 1, 2)      # (B, D, H, W)

    # 4) pooling adaptatif vers 4x8 = 32 régions
    pooled = F.adaptive_avg_pool2d(patches, (4, 8))  # (B, D, 4, 8)
    pooled = pooled.view(B, D, 32).permute(0, 2, 1)  # (B, 32, D)

    # 5) concat CLS + pooled
    z = torch.cat([cls, pooled], dim=1)  # (B, 33, D)
    z = F.normalize(z, dim=-1)

    return z

def pool_clip_197_to_cls(tokens_197: torch.Tensor) -> torch.Tensor:
    """
    tokens_197 : (B, L, D) = [CLS] + N patches
    return     : (B, 1, D) = uniquement le CLS normalisé
    """
    B, L, D = tokens_197.shape

    # 1) garder uniquement le CLS
    cls = tokens_197[:, :1, :]  # (B, 1, D)

    # 2) normaliser
    z = F.normalize(cls, dim=-1)  # (B, 1, D)

    return z

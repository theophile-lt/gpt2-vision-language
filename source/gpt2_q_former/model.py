import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect 

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


# We take directly our gpt2 decoder without modifying it 

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


class QFormerLayer(nn.Module):
    
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
    
    def __init__(self, enc_dim, d_lm, n_heads, n_queries=2, n_layers=2, drop=0.1):
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

# We define the new entire architecture

class GPT_Caption(nn.Module):

    def __init__(
        self,
        enc_dim: int,
        lm: nn.Module,     
        m_vis_tokens: int = 8,
        use_cls_only: bool = False,
        freeze_lm: bool = True,
    ):
        super().__init__()
        self.use_cls_only = use_cls_only
        self.gpt = lm
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
        x = full_embeds
        for block in self.gpt.transformer.h:
            x = block(x)
        x = self.gpt.transformer.ln_f(x)
        logits = self.gpt.lm_head(x)
        return logits

    def forward(self, patch_tokens, input_ids, labels=None): 
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
        pos_txt = torch.arange(T_txt, device=device)
        pos_txt_embeds = self.wpe(pos_txt).unsqueeze(0)      
        txt_embeds = txt_embeds + pos_txt_embeds     
        full_embeds = torch.cat([img_embeds, txt_embeds], dim=1)  

        logits = self._decode_transformer(full_embeds)
        loss = None
        if labels is not None:
            logits_text = logits[:, M:M+T_txt, :] 
            loss = F.cross_entropy(
                logits_text.reshape(-1, logits_text.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
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

    B, L, D = tokens_197.shape
    cls = tokens_197[:, :1, :]            
    patches = tokens_197[:, 1:, :]   
    N = patches.size(1)
    side = int(round(N ** 0.5))
    assert side * side == N, f"Expected square grid, got N={N}"
    patches = patches.view(B, side, side, D)   
    patches = patches.permute(0, 3, 1, 2)      
    pooled = F.adaptive_avg_pool2d(patches, (4, 8))  
    pooled = pooled.view(B, D, 32).permute(0, 2, 1)  
    z = torch.cat([cls, pooled], dim=1)
    z = F.normalize(z, dim=-1)
    return z

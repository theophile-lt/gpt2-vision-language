from dataclasses import dataclass
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module) :
     
    def __init__(self, config) :
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x) :
        B, T, C = x.size() 
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head, C// self.n_head).transpose(1,2) 
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module) :

    def __init__(self, config) :
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, z) :
        B, T, C = x.size() 
        S=z.size(1) 
        q = self.q_proj(x) # (B, T, C)
        kv = self.kv_proj(z)   # (B, S, 2C)
        k,v=kv.split(self.n_embd,dim=2)  # (B, S, C), (B, S, C)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) #(B, nh, T, hs=d_k)
        k = k.view(B,S,self.n_head, C// self.n_head).transpose(1,2) #(B, nh, S, hs)
        v = v.view(B,S,self.n_head,C//self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)        
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y



class MLP(nn.Module) :

    def __init__(self,config) :
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
            self.gelu = nn.GELU(approximate='tanh')
            self.c_proj = nn.Linear(4* config.n_embd, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1
         
    def forward(self, x) :
         x=self.c_fc(x)
         x = self.gelu(x)
         x = self.c_proj(x)
         return x


class Vision_projector(nn.Module) : 
    def __init__(self,config) :
        super().__init__()
        self.z_proj = nn.Linear(config.img_embd, config.n_embd)
    def forward(self,z) :
        z=self.z_proj(z)
        return z
    

class Block(nn.Module) :
     
    def __init__(self, config) :
        super().__init__()
        self.ln_x = nn.LayerNorm(config.n_embd)
        self.xattn = CrossAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.cross_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, z ) :
        if z is not None:
            x = x + torch.tanh(self.cross_gate) * self.xattn(self.ln_x(x), z)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass

class GPTConfig :
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    img_embd: int = 768

class GPT(nn.Module) :

    def __init__(self, config) :
        super().__init__()
        self.config = config 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            vis_proj = Vision_projector(config),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),   
                ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.__init_weights) 
        for p in self.parameters():
            p.requires_grad = False
        # Unfreezing only the projector, cross-attentions and cross_gates
        for p in self.transformer['vis_proj'].parameters():
            p.requires_grad = True
        for blk in self.transformer['h']:
            for p in blk.xattn.parameters():
                p.requires_grad = True
            blk.cross_gate.requires_grad = True

    def __init_weights(self, module) :
        if isinstance(module, nn.Linear) :
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT') :
                std*= (2*self.config.n_layer) ** -0.5  
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None :
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) :
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, z=None, targets=None, target_mask=None):
        B, T = idx.size() 
        assert T<= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) 
        tok_emb = self.transformer.wte(idx) 
        x = tok_emb + pos_emb
        z_proj = None
        if z is not None:
            z_proj = self.transformer.vis_proj(z)   
            z_proj = z_proj.to(dtype=x.dtype, device=x.device)
        #forward the blocks of the transformer
        for block in self.transformer.h :
            x = block(x, z_proj)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if target_mask is None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
            else:
                logits_flat = logits.view(-1, logits.size(-1))   
                targets_flat = targets.view(-1)                  
                mask_flat = target_mask.view(-1).to(logits_flat.device) 
                per_token = F.cross_entropy(
                    logits_flat,
                    targets_flat,
                    reduction="none"
                ) 
                per_token = per_token * mask_flat
                loss = per_token.sum() / mask_flat.sum().clamp_min(1)
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device) :
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups 
        decay_params = [p for n, p in param_dict.items() if p.dim()>=2]
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
        optimizer=torch.optim.AdamW(optim_groups, lr= learning_rate, betas = (0.9,0.95), eps= 1e-8, fused=use_fused)
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

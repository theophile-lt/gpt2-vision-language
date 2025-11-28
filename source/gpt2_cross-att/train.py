import os
import math
import time
import csv

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.utils.data import DataLoader

import tiktoken
import numpy as np 
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

from model import GPT, GPTConfig
from data import CocoClipFullTokensDataset, evaluate_cider, get_most_likely_row





#simple launch:


#export FW_OUT_DIR=/Data/theophile.laurent/datasets/edu_fineweb10B

# (exemple) reprendre un run existant :
# export LOG_DIR=/Data/theophile.laurent/datasets/gpt2_runs/20251026_053012/log
# python -u train_gpt2v2.py 2>&1 | tee -a "$LOG_DIR/restart_$(date +%H%M%S).out"

# new command : FW_OUT_DIR=/Data/theophile.laurent/datasets/edu_fineweb10B \
# CUDA_VISIBLE_DEVICES=0 \
# LOG_DIR=/Data/theophile.laurent/logs_multiM/gpt2_cross_full_e1/$(date +%Y%m%d_%H%M%S)/log \
# torchrun --standalone --nproc_per_node=1 gpt_multi_full.py


#run the training loop
from torch.distributed import init_process_group,destroy_process_group
#Attempt to autodetect the device
device="cpu"
if torch.cuda.is_available():
    device="cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device="mps"
print(f"using device: {device}")
# device="cpu" #OVERRIDE

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# >>> add this line ONCE here so it's correct everywhere (DDP or not):
device_type = "cuda" if str(device).startswith("cuda") else ("cpu" if device == "cpu" else "cpu")

amp_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

torch.manual_seed(1337)
if torch.cuda.is_available() :
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

RUN_HELLASWAG = False
total_batch_size = 524288  #previous : 524288 #2e19
B=128 #previous : 16 #micro batch size
T=32 #previous : T=1024 #sequence length
assert total_batch_size % (B*T*ddp_world_size) == 0, 'make sure total_batch_size is divisible by B*T*ddp_world_size'
grad_accum_steps = total_batch_size//(B*T*ddp_world_size)
if master_process :
    print(f"total desired batch size:{total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print("I am GPU",ddp_rank )
print("bye")
# import sys; sys.exit(0)

# Checkpoint path :
INIT_CKPT = "/Data/theophile.laurent/datasets/gpt2_runs/20251026_101905/log/ckpts/model_best.pt"

# COCO+CLIP caption loaders
COCO_ROOT = "/Data/theophile.laurent/datasets/coco2017"
CLIP_FEATS_DIR = "/Data/theophile.laurent/datasets/clip_feats"
CLIP_FULL_DIR  = "/Data/theophile.laurent/datasets/clip_feats_full"

train_ds = CocoClipFullTokensDataset(
    tokens_dir=os.path.join(CLIP_FULL_DIR, "train"),
    coco_root=os.path.join(COCO_ROOT, "train2017"),
    ann_path=os.path.join(COCO_ROOT, "annotations", "captions_train2017.json"),
    tokenizer=enc,
    max_len=T,
)
val_ds = CocoClipFullTokensDataset(
    tokens_dir=os.path.join(CLIP_FULL_DIR, "val"),
    coco_root=os.path.join(COCO_ROOT, "val2017"),
    ann_path=os.path.join(COCO_ROOT, "annotations", "captions_val2017.json"),
    tokenizer=enc,
    max_len=T,
)


train_loader = DataLoader(train_ds, batch_size=B, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=B, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

train_iter = iter(train_loader)


torch.set_float32_matmul_precision('high')

# create model
config= GPTConfig(vocab_size=50304, block_size=1024 )
model = GPT(config)


# Load checkpoint
if INIT_CKPT is not None:
    ckpt = torch.load(INIT_CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if master_process:
        print(f"[init] loaded LM weights from {INIT_CKPT}")
        print(f"[init] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

model.to(device)
if device_type == "cuda":
    model = model.to(torch.bfloat16)

use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model=DDP(model, device_ids=[ddp_local_rank]) #DistributedDataParallel
raw_model = model.module if ddp else model #always contains the "raw" unwrapped model

if master_process:
    n_train = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in raw_model.parameters())
    print(f"[init] trainable params: {n_train}/{n_total}")

# ----------------- FLOPS / PARAMS (fvcore) -----------------
if master_process:
    print("==== Parameter count (fvcore) ====")
    print(parameter_count_table(raw_model))

    # --- modèle "shadow" CPU full-float32 juste pour le comptage FLOPs ---
    flops_config = GPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        img_embd=config.img_embd,
    )
    flops_model = GPT(flops_config)          # nouveau modèle, jamais casté en bf16
    flops_model.eval()

    B_flop, T_flop = 1, T                    # même T que ton training
    x_flop = torch.randint(
        low=0,
        high=flops_config.vocab_size,
        size=(B_flop, T_flop),
        dtype=torch.long,                    # tokens en long
        device="cpu",
    )
    S_flop = 197                             # nb de tokens visuels (CLS+patches)
    z_flop = torch.randn(
        B_flop, S_flop, flops_config.img_embd,
        dtype=torch.float32,
        device="cpu",
    )

    with torch.no_grad():
        flops = FlopCountAnalysis(flops_model, (x_flop, z_flop))
        print(f"==== FLOPs (one forward, B={B_flop}, T={T_flop}) ====")
        print(f"Total FLOPs: {flops.total():.3e}")
        print(flop_count_table(flops))


max_lr = 1e-3            
min_lr = max_lr * 0.01    
warmup_steps = 20        
max_steps = 80  #One epoch = 73 steps #original : 19073       # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it) :
    # 1) linear warmup for warmup iters :
    if it < warmup_steps :
        return max_lr * (it+1) / warmup_steps
    #if it > lr_decay_iter, return min learning rate
    if it > max_steps :
        return min_lr
    # 3) in between, use cosine decay
    decay_ratio = (it - warmup_steps)/(max_steps-warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * ( 1.0 + math.cos(math.pi * decay_ratio)) #coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#Optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device_type)

# --- LOGS & CHECKPOINTS ------------
log_dir = os.environ.get("LOG_DIR", "log")
if master_process:
    print(f"[logs] writing to: {os.path.abspath(log_dir)}")
os.makedirs(log_dir, exist_ok=True)

# un petit log texte si tu veux continuer d’y écrire
log_file = os.path.join(log_dir, "log.txt")
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        pass

# timestamp pour nommer proprement CSV/XLSX et un dossier de checkpoints isolé
ts = time.strftime("%Y%m%d_%H%M%S")
csv_log = os.path.join(log_dir, f"train_{ts}.csv")
avg_dt = None
last_val_loss = None

# entête CSV une seule fois (maître uniquement)
if master_process and not os.path.exists(csv_log):
    with open(csv_log, "w", newline="") as f:
        csv.writer(f).writerow(
            ["time","phase","step","loss","lr","grad_norm","dt_ms","tok_per_s","hellaswag_acc"]
        )

# dossier de checkpoints
CKPT_DIR = os.path.join(log_dir, "ckpts")  # dossier fixe pour reprendre
if master_process:
    os.makedirs(CKPT_DIR, exist_ok=True)

best_val = float("inf")  # pour sauver le meilleur modèle
best_step = 0
best_path = os.path.join(CKPT_DIR, "model_best.pt")
SAVE_EVERY = 2500        # checkpoint périodique (ajuste si tu veux)


# --- Auto-resume depuis le dernier checkpoint "rolling" s'il existe ---
last_path = os.path.join(CKPT_DIR, "model_last.pt")
start_step = 0
if master_process:
    print(f"[ckpt] rolling path: {os.path.abspath(last_path)}")

if os.path.isfile(last_path):
    ckpt = torch.load(last_path, map_location=device)
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = int(ckpt.get("step", 0)) + 1
    if master_process:
        print(f"[ckpt] resumed from {last_path} at step {start_step}")

# >>> ajouter ceci pour DDP :
if ddp:
    dist.barrier()


#-------------------------------------------

for step in range(start_step, max_steps) :
    t0=time.time()
    last_step = (step == max_steps -1)

    #once in a while evaluate our validation loss
    if step % 20 == 0 or last_step :
        model.eval()
        with torch.no_grad():
            val_loss_accum = torch.zeros(1, device=device)
            last_val_loss = val_loss_accum.item()
            val_loss_steps = 20
            for i, (x, y, m, z) in enumerate(val_loader):
                if i >= val_loss_steps:
                    break
                x = x.to(device); y = y.to(device); m = m.to(device); z = z.to(device)
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = model(x, z=z, targets=y, target_mask=m)
                val_loss_accum += loss.detach()

            val_loss_accum /= val_loss_steps

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            last_val_loss = val_loss_accum.item()

            #-----------LOGS----------------
            # --- CSV: ligne 'val' ---
            with open(csv_log, "a", newline="") as f:
                csv.writer(f).writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"), "val", step,
                    f"{val_loss_accum.item():.6f}", "", "", "", "", ""
                ])

            # --- Checkpoints périodiques & 'best' ---
            # 1) périodique (rolling, avec optimizer) : réécrit model_last.pt à chaque fois
            if step > 0 and (step % SAVE_EVERY == 0 or last_step):
                tmp_path = os.path.join(CKPT_DIR, f".model_last_step_{step:06d}.tmp")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": getattr(raw_model, "config", None),
                    "step": step,
                    "val_loss": float(val_loss_accum.item()),
                    "ddp_world_size": ddp_world_size,
                    "ts": ts,
                }
                torch.save(checkpoint, tmp_path)
                # rename atomique pour éviter un last corrompu en cas de kill au milieu de l'écriture
                os.replace(tmp_path, last_path)
                print(f"[ckpt] rolling last: {last_path}")

            # 2) 'best' selon val_loss
            if val_loss_accum.item() < best_val:
                best_val = val_loss_accum.item()
                best_path = os.path.join(CKPT_DIR, "model_best.pt")
                best_step=step
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": getattr(raw_model, "config", None),
                    "step": step,
                    "val_loss": best_val,
                    "ddp_world_size": ddp_world_size,
                    "ts": ts,
                }
                torch.save(checkpoint, best_path)
                print(f"[ckpt] best (val_loss={best_val:.4f}): {best_path}")

            # ----------------- CIDEr eval every 50 steps -----------------
            if master_process and (step % 20 == 0 or last_step):
                try:
                    cider_score = evaluate_cider( raw_model, device, enc, COCO_ROOT,  CLIP_FULL_DIR, max_samples=500, max_new_tokens=24,)
                    print(f"[CIDEr] step {step}: {cider_score:.4f}")

                    # log CSV (phase 'cider')
                    with open(csv_log, "a", newline="") as f:
                        csv.writer(f).writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            "cider",
                            step,
                            "", "", "", "", "", f"{cider_score:.6f}",
                        ])
                except Exception as e:
                    print(f"[CIDEr] evaluation failed at step {step}: {e}")
    # -------------------------------------------------------------

            #-------------------END LOGs

    # once in a while evaluate hellaswag
    if RUN_HELLASWAG :
        if (step % 250 == 0 or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=amp_dtype):
                        logits, loss = model(tokens, z=None)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

                # --- CSV: ligne 'hella' ---
                with open(csv_log, "a", newline="") as f:
                    csv.writer(f).writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"), "hella", step,
                        "", "", "", "", "", f"{acc_norm:.4f}"
                    ])

    
    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits, loss = model(xgen, z = None) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    #training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        try:
            x, y, m, z = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y, m, z = next(train_iter)
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        z = z.to(device)
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            logits, loss = model(x, z=z, targets=y, target_mask=m)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp :
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups :
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1=time.time()
    dt=(t1-t0)
    tokens_processed = B * T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed/dt
    # time per step
    if avg_dt is None:
        avg_dt = dt
    else:
        avg_dt = 0.9 * avg_dt + 0.1 * dt
    steps_left = max_steps - step - 1
    eta_sec = steps_left * avg_dt
    eta_h = int(eta_sec // 3600)
    eta_m = int((eta_sec % 3600) // 60)
    eta_s = int(eta_sec % 60)
    #time diff in millidseconds
    if master_process:
        val_str = f"{last_val_loss:.4f}" if last_val_loss is not None else "n/a"
        eta_str = f"{eta_h:02d}h{eta_m:02d}m{eta_s:02d}s" if avg_dt is not None else "N/A"
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | val_loss: {val_str} | lr {lr:.4e} | norm: {norm.item():.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | | ETA: {eta_str}")
        
        #---------------LOGS-----------------------------------
        # --- CSV: ligne 'train' ---
        with open(csv_log, "a", newline="") as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), "train", step,
                f"{loss_accum.item():.6f}", f"{lr:.6e}", f"{norm.item():.4f}",
                f"{dt*1000:.2f}", f"{tokens_per_sec:.2f}", ""
            ])
        #----------------------------------------

# --- Checkpoint final + conversion CSV→XLSX ---
if master_process:
    # 1) ne pas re-sauver, juste rappeler le best
    try:
        print(f"[ckpt] best: {best_path}  | Best step: {best_step}  | Best val loss: {best_val:.4f}")
    except NameError:
        print("[ckpt] attention: aucun 'best' n'a été enregistré (best_path/best_step/best_val non définis).")

    # 2) CSV -> XLSX (pour Excel)
    try:
        import pandas as pd
        xlsx_path = csv_log.replace(".csv", ".xlsx")
        df = pd.read_csv(csv_log)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="metrics")
        print(f"[excel] écrit: {xlsx_path}")
    except Exception as e:
        print(f"[excel] échec conversion CSV→XLSX: {e}")




if ddp:
    destroy_process_group()















import sys;sys.exit(0)



#Prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30

tokens = enc.encode( "Hello, I'm a language model,")
tokens = torch.tensor(tokens,dtype=torch.long) #(8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)# (5,8)
x= tokens.to(device)  

#generate ! right now x is (B,T) where B=5, T=8

torch.manual_seed(42)
if device_type == "cuda":
    torch.cuda.manual_seed(42)
while x.size(1) < max_length :
    with torch.no_grad() :
        logits = model(x, z = None)
        logits = logits[:,-1,:]
        probs=F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  #keep to 50 probabiities, avoid rare tokens 
        ix= torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices,-1,ix)
        x=torch.cat((x,xcol),dim=1)

#print the generated text
for i in range(num_return_sequences) :
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

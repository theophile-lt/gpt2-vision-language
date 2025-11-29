import torch
from torch.nn import functional as F
import math
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import csv, time
from torch.utils.data import DataLoader
from model_BLIP import GPTConfig, GPT_previous, GPT_Caption, pool_clip_197_to_33_avg_with_cls
from data import CocoClipFullTokensDataset, evaluate_cider
import tiktoken



from torch.distributed import init_process_group,destroy_process_group
device="cpu"
if torch.cuda.is_available():
    device="cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device="mps"
print(f"using device: {device}")

ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


device_type = "cuda" if str(device).startswith("cuda") else ("cpu" if device == "cpu" else "cpu")
amp_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
torch.set_float32_matmul_precision('high')



enc = tiktoken.get_encoding("gpt2")


total_batch_size = 524288  
B=128 # Batch_size
T=32  # Sequence length for text
assert total_batch_size % (B*T*ddp_world_size) == 0, 'make sure total_batch_size is divisible by B*T*ddp_world_size'
grad_accum_steps = total_batch_size//(B*T*ddp_world_size)
if master_process :
    print(f"total desired batch size:{total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


# Checkpoint path :
INIT_CKPT = "/Data/theophile.laurent/datasets/gpt2_runs/20251026_101905/log/ckpts/model_best.pt"
# COCO+CLIP caption loaders
COCO_ROOT = "/Data/theophile.laurent/datasets/coco2017"
CLIP_FULL_DIR  = "/Data/theophile.laurent/datasets/clip_feats_full"


# Dataloader :

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



# Create model

config = GPTConfig(vocab_size=50304, block_size=1024)
lm = GPT_previous(config)

ckpt = torch.load(INIT_CKPT, map_location="cpu", weights_only=False)
lm.load_state_dict(ckpt["model"], strict=False)

model = GPT_Caption(
    enc_dim=768,            
    lm=lm,                  
    m_vis_tokens=32,
    use_cls_only=False,
    freeze_lm=True,
)

model.to(device)
if device_type == "cuda":
    model = model.to(torch.bfloat16)

use_compile = False 
if use_compile:
    model = torch.compile(model)
if ddp:
    model=DDP(model, device_ids=[ddp_local_rank]) 
raw_model = model.module if ddp else model 

if master_process:
    n_train = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in raw_model.parameters())
    print(f"[init] trainable params: {n_train}/{n_total}")



max_lr = 1e-3           
min_lr = 1e-4    
warmup_steps = 5
max_steps = 80  
def get_lr(it) : # Cosine decay with warmup
    if it < warmup_steps :
        return max_lr * (it+1) / warmup_steps
    if it > max_steps :
        return min_lr
    decay_ratio = (it - warmup_steps)/(max_steps-warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * ( 1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device_type)

# Logs and checkpoints
log_dir = os.environ.get("LOG_DIR", "log")
if master_process:
    print(f"[logs] writing to: {os.path.abspath(log_dir)}")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "log.txt")
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        pass

ts = time.strftime("%Y%m%d_%H%M%S")
csv_log = os.path.join(log_dir, f"train_{ts}.csv")
avg_dt = None
last_val_loss = None

if master_process and not os.path.exists(csv_log):
    with open(csv_log, "w", newline="") as f:
        csv.writer(f).writerow(
            ["time","phase","step","loss","lr","grad_norm","dt_ms","tok_per_s","hellaswag_acc"]
        )

CKPT_DIR = os.path.join(log_dir, "ckpts")  
if master_process:
    os.makedirs(CKPT_DIR, exist_ok=True)

best_val = float("inf")
best_step = 0
best_path = os.path.join(CKPT_DIR, "model_best.pt")
SAVE_EVERY = 2500        
last_path = os.path.join(CKPT_DIR, "model_last.pt")
start_step = 0


def save_rolling_checkpoint(step, val_loss):
    if not master_process:
        return
    tmp_path = os.path.join(CKPT_DIR, f".model_last_step_{step:06d}.tmp")
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": getattr(raw_model, "config", None),
        "step": step,
        "val_loss": float(val_loss),
        "ddp_world_size": ddp_world_size,
        "ts": ts,
    }
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, last_path)
    print(f"[ckpt] rolling last: {last_path}")

def save_best_checkpoint(step, val_loss):
    global best_val, best_step
    if not master_process:
        return
    if val_loss < best_val:
        best_val = val_loss
        best_step = step
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

def run_validation_and_logging(step, last_step):
    global last_val_loss
    model.eval()
    with torch.no_grad():
        val_loss_accum = torch.zeros(1, device=device)
        last_val_loss = val_loss_accum.item()
        val_loss_steps = 20
        for i, (x, y, m, z) in enumerate(val_loader):
            if i >= val_loss_steps:
                break
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            z = z.to(device)
            labels = y.clone()
            labels = labels.masked_fill(~m, -100)
            z = pool_clip_197_to_33_avg_with_cls(z)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                logits, loss = model(z, x, labels=labels)
            val_loss_accum += loss.detach()
        val_loss_accum /= val_loss_steps
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    val_loss = val_loss_accum.item()
    last_val_loss = val_loss 
    if master_process:
        print(f"validation loss: {val_loss:.4f}")
        with open(csv_log, "a", newline="") as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), "val", step,
                f"{val_loss:.6f}", "", "", "", "", ""
            ])
        if step > 0 and (step % SAVE_EVERY == 0 or last_step):
            save_rolling_checkpoint(step, val_loss)
        save_best_checkpoint(step, val_loss)
        try:
            cider_score = evaluate_cider(
                raw_model,
                device,
                enc,
                COCO_ROOT,
                CLIP_FULL_DIR,
                max_samples=500,
                max_new_tokens=24,
            )
            print(f"[CIDEr] step {step}: {cider_score:.4f}")
            with open(csv_log, "a", newline="") as f:
                csv.writer(f).writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    "cider",
                    step,
                    "", "", "", "", "", f"{cider_score:.6f}",
                ])
        except Exception as e:
            print(f"[CIDEr] evaluation failed at step {step}: {e}")
    return val_loss



if ddp:
    dist.barrier()


# Training Loop :

for step in range(start_step, max_steps) :
    t0=time.time()
    last_step = (step == max_steps -1)

    # Evaluation
    if step % 20 == 0 or last_step:
        run_validation_and_logging(step, last_step)
        
    # Training
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
        labels = y.clone()
        labels = labels.masked_fill(~m, -100)
        z = pool_clip_197_to_33_avg_with_cls(z)
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            logits, loss = model(z, x, labels=labels)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp :
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups :
        param_group['lr'] = lr
    optimizer.step()

    # Logs and time
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1=time.time()
    dt=(t1-t0)
    tokens_processed = B * T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed/dt
    if avg_dt is None:
        avg_dt = dt
    else:
        avg_dt = 0.9 * avg_dt + 0.1 * dt
    steps_left = max_steps - step - 1
    eta_sec = steps_left * avg_dt
    eta_h = int(eta_sec // 3600)
    eta_m = int((eta_sec % 3600) // 60)
    eta_s = int(eta_sec % 60)
    if master_process:
        val_str = f"{last_val_loss:.4f}" if last_val_loss is not None else "n/a"
        eta_str = f"{eta_h:02d}h{eta_m:02d}m{eta_s:02d}s" if avg_dt is not None else "N/A"
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | val_loss: {val_str} | lr {lr:.4e} | norm: {norm.item():.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | | ETA: {eta_str}")
        with open(csv_log, "a", newline="") as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), "train", step,
                f"{loss_accum.item():.6f}", f"{lr:.6e}", f"{norm.item():.4f}",
                f"{dt*1000:.2f}", f"{tokens_per_sec:.2f}", ""
            ])


# Final checkpoint
if master_process:
    try:
        print(f"[ckpt] best: {best_path}  | Best step: {best_step}  | Best val loss: {best_val:.4f}")
    except NameError:
        print("best_path/best_step/best_val not definite")
    try:
        import pandas as pd
        xlsx_path = csv_log.replace(".csv", ".xlsx")
        df = pd.read_csv(csv_log)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="metrics")
        print(f"[excel] écrit: {xlsx_path}")
    except Exception as e:
        print(f"fail conversion to xlsx: {e}")




if ddp:
    destroy_process_group()

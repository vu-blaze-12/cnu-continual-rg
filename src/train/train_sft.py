import argparse
import os
import sys
import re

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import time
try:
    import psutil
except Exception:
    psutil = None

try:
    import wandb
except ImportError:
    wandb = None

from models.baseline import load_baseline_model
from models.cnu_transformer import load_cnu_model
from data.rg_dataset import make_rg_split
from data.tokenization import get_tokenizer, TextDataset, CausalLMDataCollator
from reasoning_gym import get_score_answer_fn


def get_gpu_utilization():
    """Get GPU utilization percentage (best-effort)."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
    except Exception:
        pass
    return 0


def get_gpu_memory_stats():
    """Get GPU memory stats in GB (best-effort)."""
    try:
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            return {"allocated_gb": mem_alloc, "reserved_gb": mem_reserved}
    except Exception:
        pass
    return {"allocated_gb": 0, "reserved_gb": 0}


def log_memory():
    """Log GPU and host memory usage in GB (best-effort)."""
    try:
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated()
            mem_reserved = torch.cuda.memory_reserved()
            max_alloc = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            print(f"[MEM] CUDA allocated={mem_alloc/1e9:.3f}GB reserved={mem_reserved/1e9:.3f}GB max_alloc={max_alloc/1e9:.3f}GB max_reserved={max_reserved/1e9:.3f}GB")

        # Host memory
        if psutil is not None:
            vm = psutil.virtual_memory()
            print(f"[MEM] Host RAM total={vm.total/1e9:.3f}GB available={vm.available/1e9:.3f}GB used={vm.used/1e9:.3f}GB")

    except Exception as e:
        print(f"[MEM] Failed to collect memory stats: {e}")



# -------------------------
# Evaluation
# -------------------------
def evaluate(model, tokenizer, val_prompts, val_answers, verifier, task, device, max_new_tokens=16):
    model.eval()
    correct = 0

    for prompt, true_answer in zip(val_prompts, val_answers):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # ---- clean answer ----
        answer = output.split("Answer:")[-1]
        answer = re.sub(r"</s>.*", "", answer) 
        answer = answer.strip().split("\n")[0].strip()

        print("Generated:", repr(answer))
        print("Expected :", repr(true_answer))

        if task == "copy_task":
            score = verifier(answer, true_answer)
        else:
            score = verifier(answer, {"answer": true_answer})

        correct += score

    return correct / len(val_answers)


# -------------------------
# Training
# -------------------------
def train(args):
    # Initialize wandb
    if wandb is not None:
        wandb.init(
            project="cnu-continual-rg",
            name=f"{args.model_type}_{args.task}",
            config=vars(args)
        )
        print("✓ Wandb initialized")
    
    # Simple device logic: use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load data ===
    split = make_rg_split(args.task, n_train=args.n_train, n_val=args.n_val, seed=42)
    tokenizer = get_tokenizer()

    train_dataset = TextDataset(split["train_texts"], tokenizer)
    val_prompts = split["val_prompts"]
    val_answers = split["val_answers"]

    # === Verifier ===
    if args.task == "copy_task":
        def verifier(pred, true):
            return float(pred.strip() == true.strip())
    else:
        verifier = get_score_answer_fn(args.task)

    collator = CausalLMDataCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator
    )

    # === Load model ===
    if args.model_type == "baseline":
        model, _ = load_baseline_model()

    elif args.model_type == "cnu":
        cnu_config = {
            "key_mem_units": 4,
            "psi_fn": "resize1d",
            "key_size": 768,
            "upd_m": None,       
            "upd_k": "ad_hoc_WTA",
            "beta_k": 0.01,
            "gamma_alpha": 10.0,
            "tau_alpha": 0.95,
            "tau_mu": 50,
            "tau_eta": 50,
            "scramble": True,
            "delta": 1
        }

        model, _ = load_cnu_model("distilgpt2", cnu_config)

    else:
        raise ValueError("Invalid model type")

    # Move model to device (GPU if available)
    model.to(device)
    
    # Enable gradient checkpointing to save memory (activations recomputed during backward)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Use mixed precision (AMP) for 2-3x memory savings on GPU
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    autocast_enabled = torch.cuda.is_available()
    if autocast_enabled:
        print("✓ Mixed precision (AMP) enabled")

    # === Training loop ===
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Run forward with mixed precision
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                outputs = model(**batch)
                loss = outputs.loss

            # Print max logit to verify training progress
            with torch.no_grad():
                max_logit = outputs.logits.max().item()
                pbar.set_postfix(loss=loss.item(), max_logit=max_logit)

            # Backward with mixed precision scaling
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()

            total_loss += loss.item()

            # Log to wandb every step
            if wandb is not None:
                gpu_mem = get_gpu_memory_stats()
                wandb.log({
                    "train_loss": loss.item(),
                    "max_logits": max_logit,
                    "gpu_utilization_%": get_gpu_utilization(),
                    "gpu_allocated_gb": gpu_mem["allocated_gb"],
                    "gpu_reserved_gb": gpu_mem["reserved_gb"],
                    "epoch": epoch + 1,
                    "step": step
                })

            # Periodic memory logging
            if getattr(args, "log_interval", 0) and (step % args.log_interval == 0):
                print(f"[Epoch {epoch+1}] Step {step}")

        avg_loss = total_loss / len(train_loader)

        acc = evaluate(
            model,
            tokenizer,
            val_prompts,
            val_answers,
            verifier,
            task=args.task,
            device=device
        )

        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")

        # Log epoch metrics to wandb
        if wandb is not None:
            wandb.log({
                "epoch_train_loss": avg_loss,
                "val_accuracy": acc,
                "epoch": epoch + 1
            })

    # === Save ===
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Saved to {args.output_dir}")
    
    if wandb is not None:
        wandb.finish()


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="leg_counting")
    parser.add_argument("--model_type", choices=["baseline", "cnu"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=100)      # Quick sanity config
    parser.add_argument("--n_val", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=50, help="Number of steps between memory logs (0 to disable)")
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()
    train(args)

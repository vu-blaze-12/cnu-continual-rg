import argparse
import os
import sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from models.baseline import load_baseline_model
from models.cnu_transformer import load_cnu_model
from data.rg_dataset import make_rg_split
from data.tokenization import get_tokenizer, TextDataset, CausalLMDataCollator
from reasoning_gym import get_score_answer_fn


def evaluate(model, tokenizer, val_prompts, val_answers, verifier, max_new_tokens=16):
    model.eval()
    correct = 0

    for prompt, true_answer in zip(val_prompts, val_answers):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # extracting just the answer part
        answer = output.split("Answer:")[-1].strip().split("\n")[0].strip()
        print("Generated:", repr(answer))
        print("Expected :", repr(true_answer))

        score = verifier(answer, {"answer": true_answer})
        correct += score

    return correct / len(val_answers)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load data ===
    split = make_rg_split(args.task, n_train=args.n_train, n_val=args.n_val, seed=42)
    tokenizer = get_tokenizer()
    train_dataset = TextDataset(split["train_texts"], tokenizer)
    val_prompts = split["val_prompts"]
    val_answers = split["val_answers"]
    verifier = get_score_answer_fn(args.task)

    collator = CausalLMDataCollator(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # === Load model ===
    if args.model_type == "baseline":
        model, _ = load_baseline_model()
    elif args.model_type == "cnu":
        cnu_config = {
            "key_mem_units": 10,
            "psi_fn": "resize1d",
            "key_size": 768,
            "upd_m": "WTA",
            "upd_k": "ad_hoc_WTA",
            "beta_k": 0.01,
            "gamma_alpha": 25.0,
            "tau_alpha": 0.95,
            "tau_mu": 50,
            "tau_eta": 50,
            "scramble": True,
            "delta": 2
        }
        model, _ = load_cnu_model("distilgpt2", cnu_config)
    else:
        raise ValueError("Invalid model type")

    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # === Training loop ===
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, tokenizer, val_prompts, val_answers, verifier)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="leg_counting")
    parser.add_argument("--model_type", choices=["baseline", "cnu"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)

    args = parser.parse_args()
    train(args)

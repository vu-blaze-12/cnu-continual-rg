# src/models/smoke_test.py

from baseline import load_baseline_model
from cnu_transformer import load_cnu_model

import torch

prompt = "Question: How many legs does a spider have?\nAnswer:"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading baseline model...")
baseline_model, tokenizer = load_baseline_model()
baseline_model.to(device).eval()

print("Loading CNU model...")
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
cnu_model, _ = load_cnu_model("distilgpt2", cnu_config)
cnu_model.to(device).eval()

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# === FORWARD TEST ===
with torch.no_grad():
    print("\nBaseline forward pass...")
    out_base = baseline_model(**inputs)
    print("Baseline logits:", out_base.logits.shape)

    print("\nCNU forward pass...")
    out_cnu = cnu_model(**inputs)
    print("CNU logits:", out_cnu.logits.shape)

# === GENERATION TEST ===
print("\nBaseline generation:")
gen_base = baseline_model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(gen_base[0]))

print("\nCNU generation:")
gen_cnu = cnu_model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(gen_cnu[0]))

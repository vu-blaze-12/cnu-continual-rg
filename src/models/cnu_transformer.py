import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, "external", "continual_neural_unit")
for path in (PROJECT_ROOT, EXTERNAL_ROOT):
    if path not in sys.path:
        sys.path.append(path)

from transformers import AutoModelForCausalLM, AutoTokenizer
from external.continual_neural_unit.cnu.layers import Linear as CNULinear
import torch.nn as nn


class CNUAdapter(nn.Module):
    """
    Wrapping CNULinear to handle 3D tensors from Transformer blocks.
    """
    def __init__(self, cnu_layer):
        super().__init__()
        self.cnu = cnu_layer

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(batch_size * seq_len, hidden_dim)
        out_flat = self.cnu(x_flat)
        return out_flat.view(batch_size, seq_len, -1)


def replace_c_proj_with_cnu(model, cnu_config):
    """
    Replacing all transformer FFN output projections with CNULinear layers.
    """
    for name, module in model.named_modules():
        if "mlp.c_proj" in name:
            parts = name.split(".")
            submodule = model
            for p in parts[:-1]:
                submodule = getattr(submodule, p)

            old_proj = getattr(submodule, parts[-1])
            in_dim, out_dim = old_proj.weight.shape  

            # Creating CNULinear layer
            cnu_layer = CNULinear(
                in_features=in_dim,          # 3072
                out_features=out_dim,        # 768
                bias=True,
                shared_keys=True,

                psi_fn="resize1d",           # maps 3072 → key_size
                key_size=out_dim,            # 768

                key_mem_units=cnu_config.get("key_mem_units", 10),
                upd_m=cnu_config.get("upd_m", "WTA"),
                upd_k=cnu_config.get("upd_k", "ad_hoc_WTA"),
                beta_k=cnu_config.get("beta_k", 0.01),
                gamma_alpha=cnu_config.get("gamma_alpha", 25.0),
                tau_alpha=cnu_config.get("tau_alpha", 0.95),
                tau_mu=cnu_config.get("tau_mu", 50),
                tau_eta=cnu_config.get("tau_eta", 50),
                scramble=cnu_config.get("scramble", True),
                delta=cnu_config.get("delta", 2),
            )


            # Wrap to handle 3D tensors [B, T, D]
            setattr(submodule, parts[-1], CNUAdapter(cnu_layer))


def load_cnu_model(model_name="distilgpt2", cnu_config=None):
    """
    Load GPT-2 model and tokenizer, replacing FFN layers with CNULinear.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if cnu_config is not None:
        replace_c_proj_with_cnu(model, cnu_config)

    return model, tokenizer

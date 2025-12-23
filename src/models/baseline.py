from transformers import AutoModelForCausalLM, AutoTokenizer

def load_baseline_model(model_name="distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  #important for padding/generation

    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer

import random
import reasoning_gym as rg

def format_rg_sample(sample: dict) -> str:
    return f"Question: {sample['question'].strip()}\nAnswer: {sample['answer'].strip()}</s>"

def extract_prompt(sample: dict) -> str:
    return f"Question: {sample['question'].strip()}\nAnswer:"

def make_rg_split(task: str, n_train: int, n_val: int, seed: int = 42):
    random.seed(seed)
    dataset = rg.create_dataset(task)

    total = n_train + n_val
    all_samples = [dataset[i] for i in range(total)]
    random.shuffle(all_samples)

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:]

    return {
        "train_texts": [format_rg_sample(s) for s in train_samples],
        "val_texts":   [format_rg_sample(s) for s in val_samples],
        "val_prompts": [extract_prompt(s)   for s in val_samples],
        "val_answers": [s["answer"].strip() for s in val_samples],
        "rg_dataset": dataset
    }

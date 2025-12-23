from rg_dataset import make_rg_split
from tokenization import get_tokenizer, TextDataset, CausalLMDataCollator
from torch.utils.data import DataLoader

# Load tokenizer
tokenizer = get_tokenizer()

# Load Reasoning Gym split
split = make_rg_split("leg_counting", n_train=10, n_val=5, seed=42)

# Create Dataset and Collator
train_dataset = TextDataset(split["train_texts"], tokenizer, max_length=128)
collator = CausalLMDataCollator(tokenizer)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collator)

# Sample a batch
batch = next(iter(train_loader))

print("Batch keys:", batch.keys())
print("Input shape:", batch["input_ids"].shape)
print("Example input_ids:", batch["input_ids"][0])
print("Example labels:", batch["labels"][0])

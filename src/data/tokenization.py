from transformers import AutoTokenizer
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Dict
import torch


def get_tokenizer(model_name="distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=False,          #no padding here
            max_length=max_length,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
        }


@dataclass
class CausalLMDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  
        }

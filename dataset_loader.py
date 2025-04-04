import torch
from torch.utils.data import Dataset
from datasets import load_dataset

SYSTEM_PROMPT = "You are a math expert who explains solutions step by step."
QUESTION_TEMPLATE = "\n### Question:\n{question}\n\n### Answer:\n"


def format_gsm8k_entry(example):
    question = example["question"].strip()
    answer = example["answer"]

    if "####" in answer:
        cot, final = answer.split("####")
        final = final.strip()
        cot = cot.strip()
    else:
        cot = answer.strip()
        final = ""

    prompt = SYSTEM_PROMPT + QUESTION_TEMPLATE.format(question=question)
    target = cot + f"\nThe answer is {final}."
    return {"prompt": prompt, "target": target}


class GSM8KCoTDataset(Dataset):
    def __init__(self, split="train", tokenizer=None, max_length=512):
        raw_data = load_dataset("gsm8k", "main", split=split)
        self.data = [format_gsm8k_entry(ex) for ex in raw_data]
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer:
            self.data = [self._tokenize(ex) for ex in self.data]

    def _tokenize(self, ex):
        encoded = self.tokenizer(
            ex["prompt"], ex["target"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = encoded["input_ids"].clone()
        return encoded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

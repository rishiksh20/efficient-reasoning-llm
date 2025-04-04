import torch
from torch.utils.data import Dataset
from datasets import load_dataset

CHOICE_LETTERS = ["A", "B", "C", "D", "E"]

QUESTION_TEMPLATE = """
### Question:
{question}

Choices:
{choices}

### Answer:
"""

class CommonsenseQADataset(Dataset):
    def __init__(self, split="train", tokenizer=None, max_length=512):
        raw = load_dataset("commonsense_qa", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for ex in raw:
            question = ex["question"]
            choices = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            gold = ex["answerKey"]

            choices_text = "\n".join([f"({l}) {t}" for l, t in zip(labels, choices)])
            prompt = QUESTION_TEMPLATE.format(question=question.strip(), choices=choices_text.strip())
            target = f"The answer is ({gold})."

            if tokenizer:
                encoded = tokenizer(
                    prompt, target,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )
                encoded = {k: v.squeeze(0) for k, v in encoded.items()}
                encoded["labels"] = encoded["input_ids"].clone()
                self.data.append(encoded)
            else:
                self.data.append({"prompt": prompt, "target": target})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

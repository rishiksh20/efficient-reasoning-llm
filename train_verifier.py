# train_verifier.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def tokenize_fn(example, tokenizer, max_length=512):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)

def train_verifier(model_name="roberta-base", data_path="verifier_data.jsonl"):
    dataset = load_dataset("json", data_files=data_path)["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized = dataset.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="verifier_model",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="no",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model("verifier_model")
    print("[Verifier] Model saved to verifier_model/")

if __name__ == "__main__":
    train_verifier()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from lora import inject_lora_layers
from dataset_loader import GSM8KCoTDataset
import os


def train_lora_model(
    model_name="meta-llama/Llama-2-7b-hf",
    output_dir="checkpoints",
    epochs=3,
    batch_size=4,
    lr=2e-4,
    max_length=512,
    r=8,
    alpha=32,
    use_fp16=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = GSM8KCoTDataset(split="train", tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = inject_lora_layers(model, r=r, alpha=alpha)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=100, num_training_steps=epochs * len(loader))
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg:.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, "lora_adapter.pt"))
    print("[Train] LoRA training complete. Weights saved.")


if __name__ == "__main__":
    train_lora_model()

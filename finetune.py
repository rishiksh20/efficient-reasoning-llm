import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora import inject_lora_layers
from gptq_quantizer import quantize_model_weights
from dataset_loader import GSM8KCoTDataset
import torch.nn.functional as F

def train_lora_model():
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and quantize
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to("cuda")
    
    dataset = GSM8KCoTDataset(split="train", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = quantize_model_weights(model, dataloader)
    model = inject_lora_layers(model)

    # Freeze everything except LoRA
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    # ---- KEY FIX: force embed outputs to require grad ----
    model.get_input_embeddings().register_forward_hook(lambda m, inp, out: out.requires_grad_())
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    for epoch in range(1):
        print(f"[Epoch {epoch+1}] Starting training...")
        epoch_loss = []
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["labels"].to("cuda")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if step == 0:
                print("Loss requires_grad:", loss.requires_grad)
                print("Loss grad_fn:", loss.grad_fn)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 200 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

            epoch_loss.append(loss.item())
        print('[Epoch {epoch+1}] Average Loss: {sum(epoch_loss)/len(epoch_loss):.4f}')
        

    os.makedirs("checkpoints", exist_ok=True)
    # torch.save(model.state_dict(), "checkpoints/lora_adapter.pt")
    # Filter and save only LoRA parameters
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state_dict, "checkpoints/lora_adapter.pt")

    print("[Training complete] Adapter saved.")

if __name__ == "__main__":
    train_lora_model()

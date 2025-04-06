import os
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora import inject_lora_layers
from gptq_quantizer import quantize_model_weights, QuantizedLinear
from dataset_loader import GSM8KCoTDataset, format_gsm8k_entry


def train_lora_model(batch_size=2, epochs=1):
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to("cuda")

    # Load dataset for finetuning
    class MaskedGSM8KDataset(GSM8KCoTDataset):
        def _tokenize(self, ex):
            encoded = self.tokenizer(
                ex["prompt"], ex["target"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            encoded = {k: v.squeeze(0) for k, v in encoded.items()}
            prompt_len = len(self.tokenizer(ex["prompt"], return_tensors="pt")["input_ids"].squeeze(0))
            labels = encoded["input_ids"].clone()
            labels[:prompt_len] = -100
            encoded["labels"] = labels
            return encoded

    dataset = MaskedGSM8KDataset(split="train", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # For calibration, create prompt-only loader
    class PromptOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, examples):
            self.examples = [format_gsm8k_entry(e)["prompt"] for e in examples]
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return {
                "input_ids": tokenizer(
                    self.examples[idx],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )["input_ids"].squeeze(0)
            }

    from datasets import load_dataset
    prompt_data = load_dataset("gsm8k", "main", split="train")
    prompt_loader = DataLoader(PromptOnlyDataset(prompt_data), batch_size=2, shuffle=True)
    model = quantize_model_weights(model, prompt_loader)



    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            if hasattr(module, "fp_weight"):
                del module.fp_weight
            if hasattr(module, "scale"):
                module.scale = module.scale.half()
            if hasattr(module, "zero"):
                module.zero = module.zero.half()
            if hasattr(module, "weight"):
                del module.weight  # From original nn.Linear
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = module.bias.half()


    print("\n[Model Diagnostic] Listing all parameters and buffers with size info...\n")
    total_params = 0
    total_buffers = 0
    for name, param in model.named_parameters():
        size = param.numel() * param.element_size()
        total_params += size
        print(f"[Param] {name} | dtype={param.dtype} | size={size/1e6:.2f} MB")

    for name, buf in model.named_buffers():
        size = buf.numel() * buf.element_size()
        total_buffers += size
        print(f"[Buffer] {name} | dtype={buf.dtype} | size={size/1e6:.2f} MB")

    total = (total_params + total_buffers) / 1e6
    print(f"\n[Summary] Total parameter size: {total_params/1e6:.2f} MB")
    print(f"[Summary] Total buffer size:    {total_buffers/1e6:.2f} MB")
    print(f"[Summary] TOTAL size:           {total:.2f} MB\n")


    # assert 1==0

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/quantized_base_model.pt")
    print("[Model quantization complete]")
    
    model = inject_lora_layers(model)

    # Freeze everything except LoRA
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    # ---- KEY FIX: force embed outputs to require grad ----
    model.get_input_embeddings().register_forward_hook(lambda m, inp, out: out.requires_grad_())
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

    for epoch in range(epochs):
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

            for name, param in model.named_parameters():
                if param.requires_grad and torch.isnan(param).any():
                    print(f"[ERROR] NaN in param: {name}")

            if torch.isnan(loss) or torch.isinf(loss):
                print("[ERROR] NaN or Inf in loss!")
                exit(1)


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 500 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

            epoch_loss.append(loss.item())

        print(f'[Epoch {epoch+1}] Average Loss: {sum(epoch_loss)/len(epoch_loss):.4f}')

    
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state_dict, "checkpoints/lora_adapter.pt")
    print("[Training complete] Adapter saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and Train LoRA model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    args = parser.parse_args()

    train_lora_model(batch_size=args.batch_size, epochs=args.epochs)

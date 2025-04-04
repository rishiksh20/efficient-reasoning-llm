# import warnings
# warnings.filterwarnings('ignore')
# import torch
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from lora_wrapper import LoRAWrappedCausalLM
# from transformers import AutoTokenizer#, AutoModelForCausalLM
# from dataset_loader import GSM8KCoTDataset
# from gptq_quantizer import quantize_model_weights
# from lora import inject_lora_layers
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# def train_lora_model():
#     model_name = "meta-llama/Llama-2-7b-hf"
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
#     tokenizer.pad_token = tokenizer.eos_token

#     # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#     model = LoRAWrappedCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#     model.to("cuda")

#     dataset = GSM8KCoTDataset(split="train", tokenizer=tokenizer)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


#     model = quantize_model_weights(model, dataloader)
#     model = inject_lora_layers(model)

#     # Freeze all parameters except LoRA
#     for name, param in model.named_parameters():
#         print(name)
#         if 'lora_' not in name:
#             param.requires_grad = False
#         else:
#             param.requires_grad = True

#     total = 0
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             size = param.numel() * param.element_size() / 1e6  # in MB
#             print(f"{name}: {size:.2f} MB")
#             total += size
#     print(f"Total trainable parameters: {total:.2f} MB")

#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()

#     model.gradient_checkpointing_enable()
#     model.train()


#     optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
#     os.makedirs("checkpoints", exist_ok=True)


#     print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
#     print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

#     # print('\nFINETUNE.PY FINAL CHECK B4 TRAIN\n')
#     # for n, p in model.named_parameters():
#     #     if p.requires_grad:
#     #         print(n, p.shape)

#     for epoch in range(1):
#         print(f"[Epoch {epoch+1}] Starting training...")
#         for step, batch in enumerate(dataloader):
#             input_ids = batch["input_ids"].to("cuda").long()
#             print(">>> input_ids.requires_grad =", batch["input_ids"].requires_grad)

#             attention_mask = batch["attention_mask"].to("cuda")
#             labels = batch["labels"].to("cuda").long()

#             # with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             print('\nLOSS - ', loss, 'TYPE LOSS - ', type(loss))
#             print("\nLoss requires grad?", loss.requires_grad)
#             print("\nLoss grad fn:", loss.grad_fn)
#             logits = outputs.logits
#             print("Logits requires grad?", logits.requires_grad)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             torch.cuda.empty_cache()
#             del loss, outputs

#             if step % 10 == 0:
#                 print(f"Step {step}: Loss = {loss.item():.4f}")

#     torch.save(model.state_dict(), "checkpoints/lora_adapter.pt")
#     print("[Training Done] LoRA adapter saved.")

# if __name__ == "__main__":
#     train_lora_model()













# import torch
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from lora_wrapper import LoRAWrappedCausalLM  # <- this file contains the above class
# from dataset_loader import GSM8KCoTDataset
# from gptq_quantizer import quantize_model_weights
# from lora import inject_lora_layers
# import torch.nn.functional as F
# import os

# def train_lora_model():
#     model_name = "meta-llama/Llama-2-7b-hf"
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
#     tokenizer.pad_token = tokenizer.eos_token

#     # Load model in bf16
#     # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

#     # Load base HF model
#     hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
#     hf_model = inject_lora_layers(hf_model)
#     hf_model.gradient_checkpointing_enable()  # must call this BEFORE wrapping
#     hf_model = hf_model.to("cuda")

#     # Wrap it
#     model = LoRAWrappedCausalLM(hf_model).to("cuda")


#     # Load dataset
#     dataset = GSM8KCoTDataset(split="train", tokenizer=tokenizer)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#     # Optional: selectively quantize downstream layers (skip for now)
#     # model = quantize_model_weights(model, dataloader)

#     # Freeze all except LoRA
#     for name, param in model.named_parameters():
#         param.requires_grad = "lora" in name

#     # Optimizer
#     optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

#     # model.gradient_checkpointing_enable()
#     model.train()

#     for epoch in range(1):
#         print(f"[Epoch {epoch+1}] Starting training...")
#         for step, batch in enumerate(dataloader):
#             input_ids = batch["input_ids"].to("cuda").long()
#             attention_mask = batch["attention_mask"].to("cuda")
#             labels = batch["labels"].to("cuda").long()

#             # Manual forward + loss computation
#             # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             # logits = outputs['logits']

#             # print(">>> logits.requires_grad =", logits.requires_grad)
#             # print(">>> logits.grad_fn =", logits.grad_fn)

#             # shift_logits = logits[..., :-1, :].contiguous()
#             # shift_labels = labels[..., 1:].contiguous()

#             # loss = F.cross_entropy(
#             #     shift_logits.view(-1, shift_logits.size(-1)),
#             #     shift_labels.view(-1),
#             #     ignore_index=-100
#             # )
#             output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = output["loss"]
#             print("Loss requires_grad =", loss.requires_grad)
#             loss.backward()

#             print(f"Loss: {loss.item()} | requires_grad: {loss.requires_grad} | grad_fn: {loss.grad_fn}")

#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#     os.makedirs("checkpoints", exist_ok=True)
#     torch.save(model.state_dict(), "checkpoints/lora_adapter.pt")
#     print("[Training Done] LoRA adapter saved.")

# if __name__ == "__main__":
#     train_lora_model()













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

            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    # torch.save(model.state_dict(), "checkpoints/lora_adapter.pt")
    # Filter and save only LoRA parameters
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state_dict, "checkpoints/lora_adapter.pt")

    print("[Training complete] Adapter saved.")

if __name__ == "__main__":
    train_lora_model()

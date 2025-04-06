import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_loader import GSM8KCoTDataset, format_gsm8k_entry
from gptq_quantizer import QuantizedLinear, quantize_model_weights
from lora import inject_lora_layers, LoRALinear
from datasets import load_dataset


def evaluate_gsm8k():
    base_model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print("[Step] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    model.to("cpu")


    # # For calibration, create prompt-only loader
    # class PromptOnlyDataset(torch.utils.data.Dataset):
    #     def __init__(self, examples):
    #         self.examples = [format_gsm8k_entry(e)["prompt"] for e in examples]
    #     def __len__(self):
    #         return len(self.examples)
    #     def __getitem__(self, idx):
    #         return {
    #             "input_ids": tokenizer(
    #                 self.examples[idx],
    #                 return_tensors="pt",
    #                 truncation=True,
    #                 max_length=512,
    #                 padding="max_length"
    #             )["input_ids"].squeeze(0)
    #         }

    # from datasets import load_dataset
    # prompt_data = load_dataset("gsm8k", "main", split="train")
    # prompt_loader = DataLoader(PromptOnlyDataset(prompt_data), batch_size=2, shuffle=True)
    # model = quantize_model_weights(model, prompt_loader)




    print("[Step] Preparing prompt loader for GPTQ calibration...")
    prompt_data = load_dataset("gsm8k", "main", split="train")

    class PromptOnlyDataset(torch.utils.data.Dataset):
        def __init__(self, examples):
            self.examples = [ex["question"] for ex in examples]

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

    prompt_loader = torch.utils.data.DataLoader(PromptOnlyDataset(prompt_data), batch_size=2)

    print("[Step] Quantizing model with GPTQ...")
    model = quantize_model_weights(model, prompt_loader)

    

    print("[Step] Injecting LoRA layers...")
    model = inject_lora_layers(model)

    base_ckpt_path = "checkpoints/quantized_base_model.pt"
    adapter_path = "checkpoints/lora_adapter.pt"

    print("[Step] Loading quantized model checkpoint...")
    quant_state_dict = torch.load(base_ckpt_path, map_location="cpu")
    model.load_state_dict(quant_state_dict, strict=False)

    print("[Checkpoint] Loaded quantized model from:", base_ckpt_path)
    print("\n[Verification] Checking for QuantizedLinear layers...")
    quantized_layers = [name for name, mod in model.named_modules() if isinstance(mod, QuantizedLinear)]
    if not quantized_layers:
        print("[WARNING] No QuantizedLinear layers found!")
    for name in quantized_layers:
        print(f" - Found QuantizedLinear: {name}")

    print("[Step] Loading LoRA adapter weights...")
    lora_state_dict = torch.load(adapter_path, map_location="cpu")
    model.load_state_dict(lora_state_dict, strict=False)

    print("[Checkpoint] Loaded LoRA adapters from:", adapter_path)
    print("\n[Verification] Checking for LoRALinear layers with quant buffers...")
    lora_layers = [mod for _, mod in model.named_modules() if isinstance(mod, LoRALinear)]
    if not lora_layers:
        print("[WARNING] No LoRALinear layers found!")
    for i, mod in enumerate(lora_layers):
        print(f" - LoRALinear #{i+1}:")
        print(f"   - A shape: {mod.lora_A.shape}, norm: {mod.lora_A.data.norm():.4f}")
        print(f"   - B shape: {mod.lora_B.shape}, norm: {mod.lora_B.data.norm():.4f}")
        if mod.weight_quant is not None:
            print(f"   - Using quantized weights [shape: {mod.weight_quant.shape}]")
        else:
            print("   - [ERROR] Quantized weights missing!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    test_dataset = GSM8KCoTDataset(split="test", tokenizer=tokenizer)
    batch_size = 1

    def collate_fn(batch):
        input_ids_list = [torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch]
        attention_mask_list = [torch.tensor(sample["attention_mask"], dtype=torch.long) for sample in batch]
        max_len = max(x.size(0) for x in input_ids_list)
        padded_ids, padded_mask = [], []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - ids.size(0)
            pad_id = tokenizer.eos_token_id
            ids = torch.cat([ids, torch.tensor([pad_id]*pad_len, dtype=torch.long)]) if pad_len > 0 else ids
            mask = torch.cat([mask, torch.tensor([0]*pad_len, dtype=torch.long)]) if pad_len > 0 else mask
            padded_ids.append(ids)
            padded_mask.append(mask)
        padded_ids = torch.stack(padded_ids)
        padded_mask = torch.stack(padded_mask)
        answers = []
        if "answer" in batch[0]:
            answers = [sample["answer"] for sample in batch]
        elif "labels" in batch[0]:
            for sample in batch:
                label_ids = [tid for tid in sample["labels"] if tid != -100 and tid != tokenizer.pad_token_id]
                label_text = tokenizer.decode(label_ids, skip_special_tokens=True)
                idx = label_text.rfind("The answer is")
                answer = label_text[idx + len("The answer is"):].strip() if idx != -1 else ""
                if answer and answer[-1] in ".!?":
                    answer = answer[:-1].strip()
                answers.append(answer)
        return {"input_ids": padded_ids, "attention_mask": padded_mask, "answers": answers}

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    total, correct = 0, 0
    x = 0
    for batch in test_loader:
        x+=1
        print(f'\nBATCH {x}/{len(test_loader)}')
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        output_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
        gt_answers = batch.get("answers", [])
        for i, generated in enumerate(output_texts):
            total += 1
            idx = generated.rfind("The answer is")
            pred = generated[idx + len("The answer is"):].strip() if idx != -1 else ""
            if pred and pred[-1] in ".!?":
                pred = pred[:-1].strip()
            if pred and '###' in pred:
                pred = pred.split('###')[0]
            if pred and pred[-1] in ".!?":
                pred = pred[:-1].strip()
            if pred and '.' in pred:
                pred = pred.split('.')[0]

            true = gt_answers[i].strip()
            if true and true[-1] in ".!?":
                true = true[:-1].strip()
            if pred == true:
                correct += 1
            print(f'CORRECT ANSWER - {true} | PRED ANSWER - {pred}')
            print(f'CORRECT = {correct} | TOTAL = {total}')

    acc = correct / total if total > 0 else 0.0
    print(f"Exact Match Accuracy: {correct}/{total} = {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate_gsm8k()
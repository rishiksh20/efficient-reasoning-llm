# Efficient Reasoning LLM: Demo Notebook

# ✅ Setup
!pip install transformers datasets --quiet

# ✅ Imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_loader import GSM8KCoTDataset
from evaluate_qa import evaluate_model
from lora import inject_lora_layers
from inject_lora import merge_lora_weights
from gptq_quantizer import GPTQQuantizer

# ✅ Load tokenizer and dataset
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = GSM8KCoTDataset(split="test", tokenizer=tokenizer)

# ✅ Load and inject LoRA
model = AutoModelForCausalLM.from_pretrained(model_name)
model = inject_lora_layers(model)
model.load_state_dict(torch.load("checkpoints/lora_adapter.pt"), strict=False)

# ✅ Merge LoRA
model = merge_lora_weights(model)

# ✅ Quantize model
layer_names = [n for n, _ in model.named_modules() if any(k in n for k in ["q_proj", "k_proj", "v_proj", "o_proj"])]
quantizer = GPTQQuantizer(model, layer_names)
quantizer.collect_activations(torch.utils.data.DataLoader(dataset, batch_size=2))
quantizer.apply_quantization()
quantizer.save("checkpoints/quantized_model.pt")

# ✅ Evaluate accuracy
model.to("cuda")
evaluate_model(model, tokenizer, dataset)

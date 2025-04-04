import torch
import torch.nn as nn
from lora import LoRALinear


def merge_lora_weights(model):
    """
    Replaces LoRALinear modules with merged nn.Linear containing W0 + alpha/r * BA.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            W0 = module.weight
            A = module.lora_A
            B = module.lora_B
            alpha = module.alpha
            r = module.r
            scale = alpha / r

            delta_W = torch.matmul(B, A) * scale
            merged_weight = W0 + delta_W.to(W0.device)

            merged = nn.Linear(module.in_features, module.out_features)
            merged.weight.data = merged_weight.clone().detach()
            if module.bias is not None:
                merged.bias.data = module.bias.clone().detach()

            # Replace module in parent
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], merged)
            count += 1

    print(f"[LoRA] Merged {count} LoRA modules into Linear layers.")
    return model


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    model_name = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.load_state_dict(torch.load("lora_adapter.pt"), strict=False)

    merged_model = merge_lora_weights(model)
    torch.save(merged_model.state_dict(), "merged_lora_model.pt")
    print("[Save] Merged model saved to merged_lora_model.pt")

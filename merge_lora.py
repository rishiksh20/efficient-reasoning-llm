# merge_lora.py
import torch
from lora import LoRALinear

def merge_lora_weights_into_base(model):
    """
    Merge LoRA A @ B weights into the base weight for inference.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print(f"[Merging LoRA] {name}")
            # Extract LoRA components
            base_weight = module.weight.data
            A = module.lora_A.data  # shape: (r, in_features)
            B = module.lora_B.data  # shape: (out_features, r)
            scaling = module.scaling

            # Compute effective weight
            lora_update = (B @ A) * scaling  # shape: (out_features, in_features)
            merged_weight = base_weight + lora_update.to(base_weight.dtype)

            # Replace with standard Linear
            new_linear = torch.nn.Linear(
                in_features=module.lora_A.shape[1],
                out_features=module.lora_B.shape[0],
                bias=(module.bias is not None)
            )
            new_linear.weight.data = merged_weight.clone().to(new_linear.weight.dtype)
            if module.bias is not None:
                new_linear.bias.data = module.bias.data.clone().to(new_linear.bias.dtype)

            # Replace in model
            parent_name, attr_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, attr_name, new_linear)

    return model

if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="merged_model")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    model.load_state_dict(torch.load(args.lora_ckpt), strict=False)

    model = merge_lora_weights_into_base(model)
    model.save_pretrained(args.output_dir)
    print(f"[LoRA Merge] Merged model saved to {args.output_dir}")

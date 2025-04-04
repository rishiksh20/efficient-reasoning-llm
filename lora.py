import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    Custom LoRA module to replace nn.Linear with low-rank adaptation.
    """
    def __init__(self, original_linear, r=8, alpha=32, dropout=0.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.weight = original_linear.weight.clone().detach()
        self.bias = original_linear.bias.clone().detach() if original_linear.bias is not None else None
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.out_features, r) * 0.01)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        update = torch.nn.functional.linear(self.dropout(x), torch.matmul(self.lora_B, self.lora_A))
        return base + self.scale * update


def inject_lora_layers(model, target_keys=("q_proj", "v_proj", "k_proj", "o_proj"), r=8, alpha=32):
    """
    Replaces target linear layers in model with LoRALinear modules.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keys):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha))
            count += 1
    print(f"[LoRA] Injected LoRA into {count} layers.")
    return model

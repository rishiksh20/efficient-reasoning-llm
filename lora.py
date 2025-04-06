import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA trainable matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=torch.float32))

        # Buffers for quantized weights (injected later)
        self.register_buffer("weight_quant", None)
        self.register_buffer("scale", None)
        self.register_buffer("zero", None)

        # Optional bias
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.weight_quant is None or self.scale is None or self.zero is None:
            raise ValueError("Quantized weight, scale, and zero must be set before forward pass.")

        # Dequantize weight to x.dtype
        W = (self.weight_quant.float() - self.zero.float()) * self.scale.float()
        W = W.to(dtype=x.dtype, device=x.device)

        bias = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None

        base = torch.nn.functional.linear(x, W, bias)

        # LoRA path (in float32 for stability)
        lora_out = torch.nn.functional.linear(
            x.to(torch.float32),
            self.lora_A
        )
        lora_out = torch.nn.functional.linear(
            lora_out,
            self.lora_B
        )
        lora_out = (self.scaling * lora_out).to(dtype=base.dtype)

        return base + lora_out


def inject_lora_layers(model, r=8, alpha=16):
    for name, module in model.named_modules():
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")):
            if not hasattr(module, "weight_quant"):
                continue

            print(f"[Injecting LoRA] {name} | {type(module)}")

            lora_layer = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=r,
                alpha=alpha,
                bias=(module.bias is not None)
            )

            # Copy quantization buffers
            lora_layer.weight_quant = module.weight_quant.clone()
            lora_layer.scale = module.scale.clone()
            lora_layer.zero = module.zero.clone()

            if module.bias is not None:
                lora_layer.bias.data.copy_(module.bias.data.to(torch.bfloat16))

            # Replace in model
            parent_name, attr_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
            lora_layer = lora_layer.to(next(model.parameters()).device)
            setattr(parent, attr_name, lora_layer)

    return model


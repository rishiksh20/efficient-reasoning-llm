import torch
import torch.nn as nn
import math
from gptq_quantizer import QuantizedLinear

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False) if bias else None

        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora = torch.nn.functional.linear(
            torch.nn.functional.linear(x.to(torch.float32), self.lora_A),
            self.lora_B
        )
        lora = (self.scaling * lora).to(base.dtype)
        # print('FWD - lora grad: ',lora.requires_grad)
        # print('FWD - base grad: ',base.requires_grad)
        return base + lora

def inject_lora_layers(model, r=8, alpha=16):
    for name, module in model.named_modules():
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) and isinstance(module, (nn.Linear, QuantizedLinear)):
            print(f"[Injecting LoRA] {name} | {type(module)}")
            lora_layer = LoRALinear(
                module.in_features,
                module.out_features,
                r=r,
                alpha=alpha,
                bias=(module.bias is not None)
            )

            if hasattr(module, "fp_weight"):
                print('fp_weight')
                lora_layer.weight.data.copy_(module.fp_weight.to(torch.bfloat16))
            elif hasattr(module, "weight"):
                lora_layer.weight.data.copy_(module.weight.data.to(torch.bfloat16))
            else:
                lora_layer.weight.data.copy_(module.weight_quant.data.to(torch.bfloat16))

            if module.bias is not None:
                lora_layer.bias.data.copy_(module.bias.data.to(torch.bfloat16))

            parent_name, attr_name = name.rsplit(".", 1)
            # parent = dict(model.named_modules())[parent_name]
            # setattr(parent, attr_name, lora_layer)
            parent = dict(model.named_modules())[parent_name]
            device = next(model.parameters()).device # get original module's device
            lora_layer = lora_layer.to(device)
            setattr(parent, attr_name, lora_layer)


    return model

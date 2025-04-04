### OGG

# import torch
# import torch.nn as nn
# import math
# from gptq_quantizer import QuantizedLinear

# class LoRALinear(nn.Module):
#     def __init__(self, in_features, out_features, r=8, alpha=16, bias=True):
#         super().__init__()
#         self.r = r
#         self.alpha = alpha
#         self.scaling = alpha / r

#         # Frozen base weights (bfloat16)
#         self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16), requires_grad=False)
#         self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False) if bias else None

#         # LoRA trainable weights in float32
#         # self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))
#         # self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=torch.float32))
#         self.lora_A = nn.Parameter(torch.empty(r, in_features))
#         self.lora_B = nn.Parameter(torch.empty(out_features, r))

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in = self.weight.size(1)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)

#         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B)

#     def forward(self, x):
#         # print("[LoRA] Forward called")
#         W = self.weight.to(x.device, dtype=x.dtype)
#         b = self.bias.to(x.device, dtype=x.dtype) if self.bias is not None else None
#         A = self.lora_A.to(x.device, dtype=x.dtype)
#         B = self.lora_B.to(x.device, dtype=x.dtype)

#         base = torch.nn.functional.linear(x, W, b)
#         lora = torch.nn.functional.linear(x, self.scaling * (B @ A))
#         return base + lora



# def inject_lora_layers(model, r=8, alpha=16):
#     for name, module in model.named_modules():
#         if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) and isinstance(module, QuantizedLinear):
#             print(f"[Injecting LoRA] {name} (type={type(module)})")

#             # Create LoRA wrapper
#             lora_layer = LoRALinear(
#                 in_features=module.in_features,
#                 out_features=module.out_features,
#                 r=r,
#                 alpha=alpha,
#                 bias=module.bias is not None,
#             )
#             # lora_layer.weight.data.copy_(module.qweight.to(torch.bfloat16))  # or use your correct base weight
#             # LoRA injection inside inject_lora_layers
#             if hasattr(module, "fp_weight"):
#                 lora_layer.weight.data.copy_(module.fp_weight.to(torch.bfloat16))
#             else:
#                 raise ValueError(f"No full-precision weight found for {name}")
            
            
#             if module.bias is not None:
#                 lora_layer.bias.data.copy_(module.bias.data.to(torch.bfloat16))

#             # Inject into parent
#             parent_name = name.rsplit(".", 1)[0]
#             attr_name = name.rsplit(".", 1)[1]
#             parent = dict(model.named_modules())[parent_name]
#             setattr(parent, attr_name, lora_layer)
#     return model









### 111

# import torch
# import torch.nn as nn
# import math
# from gptq_quantizer import QuantizedLinear

# class LoRALinear(nn.Module):
#     def __init__(self, in_features, out_features, r=8, alpha=16, bias=True):
#         super().__init__()
#         self.r = r
#         self.alpha = alpha
#         self.scaling = alpha / r

#         # Frozen base weights (copied from quantized module)
#         self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16), requires_grad=False)
#         self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False) if bias else None

#         # LoRA trainable weights
#         self.lora_A = nn.Parameter(torch.empty(r, in_features), requires_grad=True)
#         self.lora_B = nn.Parameter(torch.empty(out_features, r), requires_grad=True)

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B)

#     def forward(self, x):
#         W = self.weight.to(x.device, dtype=x.dtype)
#         b = self.bias.to(x.device, dtype=x.dtype) if self.bias is not None else None
#         A = self.lora_A.to(x.device, dtype=x.dtype)
#         A.requires_grad = True
#         B = self.lora_B.to(x.device, dtype=x.dtype)
#         B.requires_grad = True
#         x.requires_grad = True
#         print("\n\nx.requires_grad =", x.requires_grad)
#         print("A.requires_grad =", A.requires_grad)
#         print("B.requires_grad =", B.requires_grad, '\n\n')

#         base = torch.nn.functional.linear(x, W, b)
#         lora = torch.nn.functional.linear(x, self.scaling * (B @ A))
#         return base + lora

# def inject_lora_layers(model, r=8, alpha=16):
#     for name, module in model.named_modules():
#         if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) and isinstance(module, QuantizedLinear):
#             print(f"[Injecting LoRA] {name} (type={type(module)})")

#             lora_layer = LoRALinear(
#                 in_features=module.in_features,
#                 out_features=module.out_features,
#                 r=r,
#                 alpha=alpha,
#                 bias=module.bias is not None,
#             )

#             # Load full-precision weight saved during quantization
#             if hasattr(module, "fp_weight"):
#                 lora_layer.weight.data.copy_(module.fp_weight.to(torch.bfloat16))
#             else:
#                 raise ValueError(f"No full-precision weight found for {name}")

#             if module.bias is not None:
#                 lora_layer.bias.data.copy_(module.bias.data.to(torch.bfloat16))

#             # Replace module in parent
#             parent_name = name.rsplit(".", 1)[0]
#             attr_name = name.rsplit(".", 1)[1]
#             parent = dict(model.named_modules())[parent_name]
#             setattr(parent, attr_name, lora_layer)

#     return model




### 222

# import torch
# import torch.nn as nn
# import math
# from gptq_quantizer import QuantizedLinear

# class LoRALinear(nn.Module):
#     def __init__(self, in_features, out_features, r=8, alpha=16, bias=True):
#         super().__init__()
#         self.r = r
#         self.alpha = alpha
#         self.scaling = alpha / r

#         self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16), requires_grad=False)
#         self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False) if bias else None

#         self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32), requires_grad=True)
#         self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=torch.float32), requires_grad=True)

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B)

#     def forward(self, x):
#         print("CHECK 1: x grad: ", x.requires_grad)
#         x = x.to(torch.float32)
#         # x.requires_grad_()
#         # Inject trivial trainable connection to bootstrap autograd
#         if not hasattr(self, "bootstrap"):
#             self.bootstrap = nn.Linear(x.size(-1), x.size(-1), bias=False).to(x.device)
#             self.bootstrap.weight.data.zero_()
#             self.bootstrap.weight.requires_grad = True

#         x = self.bootstrap(x) + x  # now x is grad-tracked

#         print(">>> [LoRA] x grad check:")
#         print("  - x.requires_grad =", x.requires_grad)
#         print("  - x.grad_fn =", x.grad_fn)
#         print("  - bootstrap.weight.requires_grad =", self.bootstrap.weight.requires_grad)

#         # Check if x is a leaf
#         print("  - x.is_leaf =", x.is_leaf)

#         W = self.weight.to(x.device, dtype=x.dtype)
#         b = self.bias.to(x.device, dtype=x.dtype) if self.bias is not None else None
#         A = self.lora_A.to(x.device, dtype=x.dtype)
#         B = self.lora_B.to(x.device, dtype=x.dtype)
#         print("\n\nx.requires_grad =", x.requires_grad)
#         print("A.requires_grad =", A.requires_grad)
#         print("B.requires_grad =", B.requires_grad, '\n\n')
#         base = torch.nn.functional.linear(x, W, b)
#         lora = torch.nn.functional.linear(x, self.scaling * (B @ A))
#         return (base + lora).to(torch.bfloat16)

# def inject_lora_layers(model, r=8, alpha=16):
#     for name, module in model.named_modules():
#         if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) and isinstance(module, nn.Linear):
#             print(f"[Injecting LoRA] {name}")
#             lora_layer = LoRALinear(module.in_features, module.out_features, r=r, alpha=alpha, bias=(module.bias is not None))

#             lora_layer.weight.data.copy_(module.weight.data.to(torch.bfloat16))
#             if module.bias is not None:
#                 lora_layer.bias.data.copy_(module.bias.data.to(torch.bfloat16))

#             parent_name = name.rsplit(".", 1)[0]
#             attr_name = name.rsplit(".", 1)[1]
#             parent = dict(model.named_modules())[parent_name]
#             setattr(parent, attr_name, lora_layer)
#     return model







### 333


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
            print(f"[Injecting LoRA] {name}")
            lora_layer = LoRALinear(
                module.in_features,
                module.out_features,
                r=r,
                alpha=alpha,
                bias=(module.bias is not None)
            )

            if hasattr(module, "fp_weight"):
                lora_layer.weight.data.copy_(module.fp_weight.to(torch.bfloat16))
            else:
                lora_layer.weight.data.copy_(module.weight.data.to(torch.bfloat16))

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

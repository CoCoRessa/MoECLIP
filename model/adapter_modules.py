from torch import nn, einsum
import ipdb
import math
import torch
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange, repeat


class SimpleAdapter(nn.Module):
    def __init__(self, c_in, c_out=768):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())

    def forward(self, x):
        x = self.fc(x)
        return x

class SimpleProj(nn.Module):
    def __init__(self, c_in, c_out=768, relu=True):
        super(SimpleProj, self).__init__()
        if relu:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_out, bias=False), 
                nn.LeakyReLU()
            )
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)
            
        self.fc.apply(self._init_weights_fn)

    def _init_weights_fn(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class SimpleLoRA(nn.Module):
    def __init__(self, c_in, c_out=768, r=8, alpha = 16, init_option="lora"):
        super(SimpleLoRA, self).__init__()
        self.r = r
        self.random_orth = False
        self.alpha = alpha
        self.scaling_ = self.alpha / self.r
        
        self.lora_A = nn.Linear(c_in, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, c_out, bias=False)
            
        if self.random_orth:
            for param in self.lora_A.parameters():
                param.requires_grad = False            
            random_matrix = torch.rand(c_in, self.r)
            q, r = torch.linalg.qr(random_matrix)
            with torch.no_grad():
                self.lora_A.weight.copy_(q.T)
            scaling_factor = 1. 
            self.lora_A.weight.data *= scaling_factor
        else:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.zeros_(self.lora_B.weight)
        else:
            raise NotImplementedError
    
    def forward(self, x):
        lora_out = self.lora_A(x)
        lora_output = self.lora_B(lora_out) * self.scaling_
        
        return lora_output
    
class ConvAdapterProj(nn.Module):
    def __init__(self, c_in, c_out=768, kernel_size=3):
        super(ConvAdapterProj, self).__init__()
        self.ln = nn.LayerNorm(c_in)
        self.adapter = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel_size=kernel_size, padding='same', groups=c_in, bias=False),
            nn.GELU(),
            nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.adapter:
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x_norm = self.ln(x)
        x_permuted = x_norm.permute(0, 2, 1)
        adapter_output = self.adapter(x_permuted)
        output = adapter_output.permute(0, 2, 1)
        
        return output
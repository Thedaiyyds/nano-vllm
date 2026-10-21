import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        #温度采样
        #1. T=1：保持原始概率分布。
        #2. T<1：概率分布更尖锐，高概率 Token 更容易被选中。
        #3. T>1：概率分布更平坦，生成结果更随机。
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        #转化为概率分布
        probs = torch.softmax(logits, dim=-1)
        #按概率进行随机采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

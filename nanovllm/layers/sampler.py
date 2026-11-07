import torch
from torch import nn


def _sample_positive(logits: torch.Tensor, temperatures: torch.Tensor):
    # Preserve the original positive-temperature arithmetic and RNG consumption.
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(
        dim=-1
    )


class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        if not torch.isfinite(temperatures).all() or (temperatures < 0).any():
            raise ValueError("temperatures must be finite and nonnegative")
        positive = temperatures > 0
        if positive.all():
            return _sample_positive(logits, temperatures)
        sample_tokens = logits.argmax(dim=-1)
        if positive.any():
            sample_tokens[positive] = _sample_positive(
                logits[positive], temperatures[positive]
            )
        return sample_tokens

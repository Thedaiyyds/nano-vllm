from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError(
                f"temperature={self.temperature!r} must be finite and nonnegative"
            )
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError(
                f"max_tokens={self.max_tokens!r} must be a positive integer"
            )

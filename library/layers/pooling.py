import torch
import torch.nn as nn


class GlobalAveragePool2D(nn.Module):

    def __init__(self, start_dim=-2, end_dim=-1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sd = len(x.shape) + self.start_dim if self.start_dim < 0 else self.start_dim
        ed = len(x.shape) + self.end_dim if self.end_dim < 0 else self.end_dim
        return torch.mean(torch.flatten(x, sd, ed), dim=sd)
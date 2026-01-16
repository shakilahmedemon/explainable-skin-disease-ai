import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

# Example usage:
# 1. Load ensemble logits on validation set
# 2. Fit TemperatureScaler to minimize NLL
# 3. Apply to test logits


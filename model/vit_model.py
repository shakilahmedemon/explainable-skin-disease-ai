import torch
import torch.nn as nn
import timm

class SkinCancerViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(x)
        return self.model(x)


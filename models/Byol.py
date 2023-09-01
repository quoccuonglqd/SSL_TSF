import copy
import torch.nn as nn
from lightly.models.utils import deactivate_requires_grad
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead

class BYOL(nn.Module):
    def __init__(self, backbone, input_size=224, hidden_size=512, projection_size=256):
        super().__init__()

        print("input_size: ", input_size)

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(input_size, hidden_size, projection_size)
        self.prediction_head = BYOLPredictionHead(projection_size, hidden_size, projection_size)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone.forward_backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum.forward_backbone(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

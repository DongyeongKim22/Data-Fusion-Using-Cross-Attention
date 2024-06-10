import torch
import torch.nn as nn
from torchvision import models

class MMFImageEncoder(nn.Module):
    def __init__(self, model_name='resnet50', pretrained = True, output_dim=512):
        super(MMFImageEncoder, self).__init__()
        self.model = models.__dict__[model_name](weights=pretrained)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.fc = nn.Linear(2048, output_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        features = self.model(images)
        features = self.global_avg_pool(features)
        features = features.view(features.size(0), -1)
        image_features = self.fc(features)
        return image_features
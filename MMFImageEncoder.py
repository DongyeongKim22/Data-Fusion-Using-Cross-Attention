import torch
import torch.nn as nn
from torchvision import models

class MMFImageEncoder(nn.Module):
    def __init__(self, model_name='resnet50', pretrained = True, output_dim=512):
        super().__init__()
        self.model = models.__dict__[model_name](weights='IMAGENET1K_V2' if pretrained else None)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.fc = nn.Linear(2048, output_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(1024, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(negative_slope=0.01),            
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
            )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        images = self.model(images)
        ##############CLIP#####################
        # images = self.conv(images)
        # images = self.global_avg_pool(images)
        # images = images.view(images.size(0), -1)
        # images = self.fc(images)
        ##########################################
        batch_size, num_channels, feat_height, feat_width = images.size()
        num_patches = feat_height * feat_width
        images = images.view(batch_size, num_channels, num_patches).permute(2, 0 ,1)
        return images
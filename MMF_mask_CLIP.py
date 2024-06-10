from collections.abc import MutableMapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from MMFtextEncoder_CLIP import MMFTextEncoder
from MMFImageEncoder_CLIP import MMFImageEncoder
from torchvision import models

class MMF_mask_CLIP(nn.Module):
    def __init__(self, d_model=768, num_heads=8, target_seq_len=256):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder = MMFTextEncoder()
        self.image_encoder = MMFImageEncoder()
        self.image_attention = nn.MultiheadAttention(d_model, num_heads)
        self.target_seq_len = target_seq_len
        # self.m = nn.Linear(768*8, 512*8)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [8, 1, 512, 512] -> [8, 32, 256, 256]
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [8, 32, 256, 256] -> [8, 64, 128, 128]
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [8, 64, 128, 128] -> [8, 128, 64, 64]
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [8, 128, 64, 64] -> [8, 64, 128, 128]
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [8, 64, 128, 128] -> [8, 32, 256, 256]
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # [8, 32, 256, 256] -> [8, 16, 512, 512]
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # [8, 16, 512, 512] -> [8, 1, 512, 512]
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.linear = nn.Sequential(
                              nn.Linear(32, 64),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(64, 64),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(64, 128),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(128, 128),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(128, 256),
                              nn.LeakyReLU(negative_slope=0.01),
                              )
    def dimension_process(self, image, text):
        batch_size = text.size(1)
        text = text.permute(1, 2, 0)
        text = nn.functional.interpolate(text, size=32, mode='linear', align_corners=False)
        text = self.linear(text)
        image = image.permute(1, 2, 0)
        image = self.global_avg_pool(image)
        text = self.global_avg_pool(text)
        image = image.permute(0, 2, 1)
        mapping = torch.bmm(text, image)
        mapping = mapping.view(mapping.size(0)*mapping.size(1), mapping.size(2))
        m = nn.Linear(768*batch_size, 512*batch_size).to(self.device)     
        mapping = m(mapping.permute(1,0))
        mapping = mapping.permute(1,0)
        return mapping

    def forward(self, image, text):
        #text list
        text = self.text_encoder(text)
        image = self.image_encoder(image)
        batch_size = text.size(1)
        mapping = self.dimension_process(image, text)
        mapping = mapping.view(batch_size, 1, 512, 512)
        mapping = self.conv(mapping)
        mapping = nn.Sigmoid()(mapping)
        return mapping
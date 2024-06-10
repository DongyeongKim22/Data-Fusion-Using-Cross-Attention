from collections.abc import MutableMapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from MMFtextEncoder import MMFTextEncoder
from MMFImageEncoder import MMFImageEncoder
from torchvision import models

    
class MMF_mask_cat(nn.Module):
    def __init__(self, d_model=768, num_heads=8, target_seq_len=256):
        super().__init__()
        self.text_encoder = MMFTextEncoder()
        self.image_encoder = MMFImageEncoder()
        self.image_attention = nn.MultiheadAttention(d_model, num_heads)
        self.target_seq_len = target_seq_len
        self.conv = nn.Sequential(
            nn.Conv2d(1536, 768, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=d_model, out_channels=512, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
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
    def forward(self, image, text):
        #text list
        text = self.text_encoder(text)
        image = self.image_encoder(image)
        batch_size = text.size(1)
        text = text.permute(1, 2, 0)
        text = nn.functional.interpolate(text, size=32, mode='linear', align_corners=False)
        text = self.linear(text)
        image = image.permute(1, 2, 0)
        
        mapping = torch.cat((image, text), dim =1)
        mapping = mapping.view(batch_size, 1536, 16, 16)
        mapping = self.conv(mapping)
        mapping = nn.Sigmoid()(mapping)

        return mapping



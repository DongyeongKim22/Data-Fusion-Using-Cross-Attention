from collections.abc import MutableMapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from MMFtextEncoder import MMFTextEncoder
from MMFImageEncoder import MMFImageEncoder
from torchvision import models
from MultiHeadDotProduct import MultiHeadDotProduct

class MMF_mask(nn.Module):
    def __init__(self, d_model=768, num_heads=8, target_seq_len=256):
        super().__init__()
        self.text_encoder = MMFTextEncoder()
        self.image_encoder = MMFImageEncoder()
        self.image_attention = nn.MultiheadAttention(d_model, num_heads)
        self.target_seq_len = target_seq_len
        self.multihead_mapping = MultiHeadDotProduct()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d_model, out_channels=512, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
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

        batch_size = text.size(0)
        maping = multihead_mapping(image, text)
        mapping = self.conv(mapping)
       
        return mapping
from collections.abc import MutableMapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from MMFtextEncoder import MMFTextEncoder
from MMFImageEncoder import MMFImageEncoder
from torchvision import models

class postprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride = 2,  padding=1),  # Convolutional layer
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride = 2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3,stride = 2,  padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsampling layer
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Output layer for classification
        )
        self.sigmoid = nn.Sigmoid()
        # self.linear = nn.Linear(d_model=512, d_model=512)
    
    def forward(self, x):
        x = self.decoder(x)
        x = self.sigmoid(x)
        return x
    
class MMF_mask_multihead(nn.Module):
    def __init__(self, d_model=2048, num_heads=1, target_seq_len=256):
        super().__init__()
        self.text_encoder = MMFTextEncoder()
        self.image_encoder = MMFImageEncoder()
        self.postprocess = postprocess()
        self.image_attention = nn.MultiheadAttention(d_model, num_heads)
        self.target_seq_len = target_seq_len
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2),
      
            nn.LeakyReLU(negative_slope=0.01),            
            nn.Conv2d(d_model, 1024, kernel_size=3, padding=1),
            
            nn.LeakyReLU(negative_slope=0.01),            
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2),
           
            nn.LeakyReLU(negative_slope=0.01),            
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2),
         
            nn.LeakyReLU(negative_slope=0.01),            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
      
            nn.LeakyReLU(negative_slope=0.01),            
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
     
            nn.LeakyReLU(negative_slope=0.01),           
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
     
            nn.LeakyReLU(negative_slope=0.01),            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.linear = nn.Sequential(
                              nn.Linear(768, 768),
                              # nn.BatchNorm2d(768),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(768, 1024),
                              # nn.BatchNorm2d(1024),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(1024, 1024),
                              # nn.BatchNorm2d(1024),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(1024, 2048),
                              # nn.BatchNorm2d(2048),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(2048, 2048),
                              # nn.BatchNorm2d(2048),
                              nn.LeakyReLU(negative_slope=0.01),
                              )
    def text_padding(self,text):
        target_seq_len = 256
        current_seq_len = text.size(0)
        pad_size = target_seq_len - current_seq_len
        text = F.pad(text, (0, 0, 0, 0, 0, pad_size))
        target_embed_size = 2048
        pad_size = target_embed_size - text.size(-1)
        padding = (0, pad_size)
        text = F.pad(text, padding, "constant", 0)
        # print(text.shape)
        return text

    def forward(self, image, text):
        #text list
        text = self.text_encoder(text)
        text = self.text_padding(text)
        
        image = self.image_encoder(image)
        batch_size = text.size(1)
        # text = self.linear(text)
        
        mapping, _ = self.image_attention(query=text, key=image, value=image)        
        # mapping = self.decoder(tgt=text, memory=image)
        mapping = mapping.permute(1, 2, 0)
        mapping = mapping.view(mapping.size(0), mapping.size(1), 16, 16)
        mapping = self.conv(mapping)
        mapping = nn.Sigmoid()(mapping)
        return mapping
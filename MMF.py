import torch
import torch.nn as nn
import torch.nn.functional as F
from MMFtextEncoder import MMFTextEncoder
from MMFImageEncoder import MMFImageEncoder
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from torchvision import models

class ImageRestorationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(pretrained=True)

        self.conv_s = nn.Sequential(
          nn.Conv2d(1, 32, kernel_size=3, padding=1),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(32, 64, kernel_size=3, padding=1),
          nn.LeakyReLU(0.2, inplace=True),
          )
        self.conv_s2 = nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=3, padding=1),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(32, 64, kernel_size=3, padding=1),
          nn.LeakyReLU(0.2, inplace=True),
          )
        self.resnet_layers = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh()
        )
        self.resnet_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh()
        )

    def forward(self, x, image):
        x = self.conv_s(x)
        image = self.conv_s2(image)
        x = torch.cat((x, image), dim=1)  # [batch, 128, 512, 512]
        # x = self.conv(x)
        x = self.resnet_layers(x)  # Pass through modified ResNet
        x = self.resnet_conv(x)
        return x
    
class MMF(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = MMFTextEncoder()
        self.image_encoder = MMFImageEncoder()
        self.image_restore = ImageRestorationNetwork()

    def text_to_image_tensor(self, text_embeddings, output_size=(16, 16)):
        batch_size, latent_dim = text_embeddings.size()
        text_embeddings = text_embeddings.unsqueeze(-1).unsqueeze(-1)
        text_embeddings = F.interpolate(text_embeddings, size=output_size, mode='bilinear')
        return text_embeddings

    def forward(self, image, text):
        #text list
        text = self.text_encoder(text)
        # text = text.view(text.size(0), text.size(1), 1, 1)
        # text = self.upsample(text)
        image_features = self.image_encoder(image)
        batch_size = text.size(0)
        
        mapping = torch.bmm(text.unsqueeze(2), image_features.unsqueeze(1))
        mapping = mapping.view(batch_size, 1, mapping.size(1), mapping.size(2))
        output = self.image_restore(mapping, image)
        # output = torch.cat((image, text), dim=1)
        # output = self.upsample2(output)
        return output
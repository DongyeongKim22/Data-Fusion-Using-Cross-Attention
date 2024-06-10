import torch
import torch.nn as nn
import torch.nn.functional as F
from MMFtextEncoder import MMFTextEncoder
from MMFImageEncoder import MMFImageEncoder
from RRDBNet_arch import RRDBNet

# ESRGAN 모델 정의
class ESRGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=64, num_blocks=10):
        super(ESRGAN, self).__init__()
        self.model = RRDBNet(in_nc=in_channels, out_nc=out_channels, nf=num_filters, nb=num_blocks)
    def forward(self, x):
        return self.model(x)

    
class MMF_esr(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = MMFTextEncoder()
        self.image_encoder = MMFImageEncoder()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.esrgan = ESRGAN().cuda()
        pretrained_weights_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'
        checkpoint = torch.load(pretrained_weights_path)
        self.esrgan.model.load_state_dict(checkpoint, strict=False)

    def forward(self, image, text):
        #text list
        text = self.text_encoder(text)
        image = self.image_encoder(image)
        batch_size = text.size(0)
        
        image = torch.bmm(text.unsqueeze(2), image.unsqueeze(1))
        image = image.view(batch_size, 1, image.size(1), image.size(2))
        image = self.conv(image)
        image = self.esrgan(image)
        return image
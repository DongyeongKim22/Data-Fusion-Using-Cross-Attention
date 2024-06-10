import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random

class MagicBrushDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        source_img = item['source_img']
        target_img = item['mask_img']

        source_img = self.transform(source_img)
        target_img= self.transform_mask(target_img)

        while source_img.shape[0] != 3 or target_img.shape[0] != 4:
          # print(f"Skipping index {idx}: source_img shape {source_img.shape} or target_img shape {target_img.shape} is invalid")
          idx = random.randint(0, len(self.dataset) - 1)
          item = self.dataset[idx]
          source_img = self.transform(item['source_img'])
          target_img = self.transform(item['mask_img'])

        target_img = target_img[-1, :, :].unsqueeze(0)

        target_img = 1 - target_img
        
        instruction = item['instruction']

        return source_img, target_img, instruction
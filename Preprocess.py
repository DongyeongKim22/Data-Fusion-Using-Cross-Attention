import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class MagicBrushDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        source_img = item['source_img']
        target_img = item['target_img']
        
        if source_img.mode != 'RGB':
            source_img = source_img.convert('RGB')
        if target_img.mode != 'RGB':
            target_img = target_img.convert('RGB')

        source_img = self.transform(source_img)
        target_img = self.transform(target_img)
        
        instruction = item['instruction']

        return source_img, target_img, instruction
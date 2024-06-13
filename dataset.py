import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        
        if mode == 'train':
            self.root = os.path.join(root, 'train')
        elif mode == 'test':
            self.root = os.path.join(root, 'test')
        else:
            raise ValueError("Invalid mode. Expected one of ['train', 'test']")
        
        self.images = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        image = Image.open(img_path).convert('RGB')  # 确保图像是RGB模式

        if self.transform:
            image = self.transform(image)

        return image,image
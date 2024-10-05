import torch
from torch.utils.data import Dataset
from PIL import Image

class MyData(Dataset):

    # initialization
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    # data length
    def __len__(self):
        return len(self.image_paths)

    # load data via idx and image preprocessing
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
import torch 
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

class MNIST(Dataset):
    def __init__(self):
        self.imgs = datasets.MNIST(
            root="data",
            train=True,
            download=False,
            transform=ToTensor()
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, label = self.imgs[idx]
        return img, label


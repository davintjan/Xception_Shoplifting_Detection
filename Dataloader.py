import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class ShopliftingDataset:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Resize to Xception input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load dataset
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

        # Create DataLoader
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Class names
        self.class_names = self.dataset.classes
        print(f"Classes: {self.class_names}")  # Should print {'Normal': 0, 'Shoplifting': 1}

    def get_dataloader(self):
        return self.loader

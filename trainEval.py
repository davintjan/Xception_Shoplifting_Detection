import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataloader import ShopliftingDataset
from tqdm import tqdm
from loadPreTrain import XceptionPretrained
import os

class Trainer:
    def __init__(self, model, train_loader, learning_rate=0.001, num_epochs=10, device=None, log_dir="runs/train_logs", save_dir="checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.num_epochs = num_epochs

        # Loss function & optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # TensorBoard Writer
        self.writer = SummaryWriter(log_dir)

        # Create directory for saving models
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def train(self):
        print(f"Training on: {self.device}")
        self.model.train()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch_idx, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

                # Log loss per batch
                self.writer.add_scalar("Loss/Batch", loss.item(), epoch * len(self.train_loader) + batch_idx)

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Avg Loss: {avg_loss}")

            # Log average loss per epoch
            self.writer.add_scalar("Loss/Epoch", avg_loss, epoch)

            # Save model after each epoch
            self.save_model(epoch)

        print("Training complete!")
        self.writer.close()

    def save_model(self, epoch):
        file_path = os.path.join(self.save_dir, f"xception_shoplifting_epoch{epoch+1}.pth")
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved at {file_path}")


if __name__ == "__main__":
    # Step 1: Load dataset
    data_dir = r"D:\Akses_ai\shoplift\Model_1\Dataset"
    dataset = ShopliftingDataset(data_dir)
    train_loader = dataset.get_dataloader()

    # Step 2: Load model
    model = XceptionPretrained()

    # Step 3: Train model
    trainer = Trainer(model, train_loader)
    trainer.train()

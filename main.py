from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from utils import get_default_device, to_device
from utils import DeviceDataLoader
import torchmetrics
from tqdm import tqdm
from multiprocessing import Lock
import time
import matplotlib.pyplot as plt

lock = Lock()
# load images to tensors
size = 224
crop = 224

path_to_imgs = "~\\Documents\\FLEET\\simpsons_dataset"
transform = transforms.Compose([transforms.Resize([size, size]), transforms.ToTensor(),
                                transforms.Normalize(std=[0.229, 0.224, 0.225],
                                                     mean=[0.485, 0.456, 0.406]),
                                ])
dataset = datasets.ImageFolder(path_to_imgs, transform=transform)

# Split data into train and valid parts
valid_size = 3000
train_size = len(dataset) - valid_size

train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# Create Batch data loader
batch_size = 48
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
valid_dl = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=4)


class ImageClassification(nn.Module):
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=43, k_func=1).to('cuda')

    def predict(self, batch):
        images, labels = batch
        tar = self(images)
        loss = nn.CrossEntropyLoss()
        loss = loss(tar, labels)
        return tar, loss

    def evaluate(self, valid_dl):
        self.eval()
        for batch in valid_dl:
            images, labels = batch
            out = self(images)
            loss_ = nn.CrossEntropyLoss()
            loss_ = loss_(out, labels)
            accuracy_e = self.accuracy(out, labels)
        return loss_.detach(), accuracy_e

    def train_step(self, train_dal, opt, pbar):
        self.train()
        running_loss = 0.
        for images, labels in tqdm(train_dal):
            opt.zero_grad()

            output = self(images)
            loss = nn.CrossEntropyLoss()
            loss = loss(output, labels)
            loss.backward()
            opt.step()
            running_loss += loss
            with torch.no_grad():
                train_loss = (running_loss / len(train_dl))
            pbar.set_description(f'Train loss for batch: {train_loss:.4f}')
            pbar.update(1)

        return train_loss.item()

    def valid_step(self, valid_datal, pbar_2):
        self.eval()
        correct_total = 0.
        running_loss = 0.
        with torch.no_grad():
            for images, labels in valid_datal:
                output = self(images)
                prediction = output.argmax(dim=1)
                correct_total += prediction.eq(labels.view_as(prediction)).sum()

                loss = nn.CrossEntropyLoss()
                loss = loss(output, labels)

                running_loss += loss
                pbar_2.set_description(f'Running loss: {running_loss:.4f}')
                pbar_2.update(1)
        valid_loss = (running_loss / len(valid_datal))
        accuracy = (correct_total / len(valid_datal))
        return valid_loss.item(), accuracy.item()

    def train_epochs(self, train_dl, valid_dl, epochs=40, optimizer=torch.optim.SGD, lr=5e-5):
        self.train()
        optimizer = optimizer(self.parameters(), lr, momentum=0.9, weight_decay=1e-4)
        train_losses = []
        valid_losses = []
        valid_accuracies = []

        pbar = tqdm(range(len(train_dl)), desc='Train loss for batch: ', leave=True)
        pbar.set_lock(lock)

        pbar_2 = tqdm(range(len(valid_dl)), desc='Running loss: ', leave=True)
        pbar_2.set_lock(lock)
        for _ in (pbar_1 := tqdm(range(epochs), leave=True, desc='Avg. train/loss: ')):

            train_loss = self.train_step(train_dl, optimizer, pbar)
            pbar.reset()

            valid_loss, valid_acc = self.valid_step(valid_dl, pbar_2)
            pbar_2.reset()

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)

            pbar_1.set_description(f'Avg. train/loss: {train_loss:.4f}/{valid_loss:.4f}')
        pbar_2.close()
        pbar.close()
        pbar_1.close()
        figure = plt.figure(figsize=(16, 12))
        plt.plot(train_losses[1:], label='train loss',)
        plt.plot(valid_losses[1:], label='valid loss')
        plt.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        figure.show()
        plt.plot(valid_accuracies)
        plt.ylim([0, 1])
        plt.show()


class SimpsonsCnnModel(ImageClassification):
    def __init__(self):
        super(SimpsonsCnnModel, self).__init__()

        self.network = nn.Sequential(
            # conv2d layer
            nn.Conv2d(3, 32, 1, padding=1, stride=2, device='cuda:0', bias=False),
            # 1 bottleneck
            nn.Conv2d(32, 32 * 1, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64 * 1, kernel_size=3, padding=1, groups=32, stride=1, device='cuda:0', bias=False),
            nn.Conv2d(64 * 1, 64 * 1, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(64 * 1),
            nn.Conv2d(64 * 1, 16, kernel_size=1, padding=1, device='cuda:0', bias=False),

            # 2 bottleneck layer
            nn.Conv2d(16, 16 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(16 * 6),
            nn.Conv2d(16 * 6, 32 * 6, kernel_size=3, padding=1, groups=16, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(32 * 6, 32 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(32 * 6),
            nn.Conv2d(32 * 6, 16, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(16, 16 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(16 * 6),
            nn.Conv2d(16 * 6, 32 * 6, kernel_size=3, padding=1, groups=16, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(32 * 6, 32 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(32 * 6),
            nn.Conv2d(32 * 6, 24, kernel_size=1, padding=1, device='cuda:0', bias=False),

            # 3 bottleneck layer
            nn.Conv2d(24, 24 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(24 * 6),
            nn.Conv2d(24 * 6, 48 * 6, kernel_size=3, padding=1, groups=24, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(48 * 6, 48 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(48 * 6),
            nn.Conv2d(48 * 6, 24, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(24, 24 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(24 * 6),
            nn.Conv2d(24 * 6, 48 * 6, kernel_size=3, padding=1, groups=24, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(48 * 6, 48 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(48 * 6),
            nn.Conv2d(48 * 6, 24, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(24, 24 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(24 * 6),
            nn.Conv2d(24 * 6, 48 * 6, kernel_size=3, padding=1, groups=24, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(48 * 6, 48 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(48 * 6),
            nn.Conv2d(48 * 6, 32, kernel_size=1, padding=1, device='cuda:0', bias=False),

            # 4 bottleneck layer
            nn.Conv2d(32, 32 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(32 * 6),
            nn.Conv2d(32 * 6, 64 * 6, kernel_size=3, padding=1, groups=32, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(64 * 6, 64 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(64 * 6),
            nn.Conv2d(64 * 6, 32, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(32, 32 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(32 * 6),
            nn.Conv2d(32 * 6, 64 * 6, kernel_size=3, padding=1, groups=32, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(64 * 6, 64 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(64 * 6),
            nn.Conv2d(64 * 6, 32, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(32, 32 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(32 * 6),
            nn.Conv2d(32 * 6, 64 * 6, kernel_size=3, padding=1, groups=32, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(64 * 6, 64 * 6, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(64 * 6),
            nn.Conv2d(64 * 6, 32, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(32, 32 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(32 * 6),
            nn.Conv2d(32 * 6, 64 * 6, kernel_size=3, padding=1, groups=32, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(64 * 6, 64 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(64 * 6),
            nn.Conv2d(64 * 6, 64, kernel_size=1, padding=1, device='cuda:0', bias=False),

            # 5 bottleneck layer
            nn.Conv2d(64, 64 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(64 * 6),
            nn.Conv2d(64 * 6, 128 * 6, kernel_size=3, padding=1, groups=64, stride=1, device='cuda:0', bias=False),
            nn.Conv2d(128 * 6, 128 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(128 * 6),
            nn.Conv2d(128 * 6, 64, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(64, 64 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(64 * 6),
            nn.Conv2d(64 * 6, 128 * 6, kernel_size=3, padding=1, groups=64, stride=1, device='cuda:0', bias=False),
            nn.Conv2d(128 * 6, 128 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(128 * 6),
            nn.Conv2d(128 * 6, 64, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(64, 64 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(64 * 6),
            nn.Conv2d(64 * 6, 128 * 6, kernel_size=3, padding=1, groups=64, stride=1, device='cuda:0', bias=False),
            nn.Conv2d(128 * 6, 128 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(128 * 6),
            nn.Conv2d(128 * 6, 96, kernel_size=1, padding=1, device='cuda:0', bias=False),

            # 6 bottleneck layer
            nn.Conv2d(96, 96 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(96 * 6),
            nn.Conv2d(96 * 6, 192 * 6, kernel_size=3, padding=1, groups=96, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(192 * 6, 192 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(192 * 6),
            nn.Conv2d(192 * 6, 96, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(96, 96 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(96 * 6),
            nn.Conv2d(96 * 6, 192 * 6, kernel_size=3, padding=1, groups=96, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(192 * 6, 192 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(192 * 6),
            nn.Conv2d(192 * 6, 96, kernel_size=1, padding=1, device='cuda:0', bias=False),

            nn.Conv2d(96, 96 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(96 * 6),
            nn.Conv2d(96 * 6, 192 * 6, kernel_size=3, padding=1, groups=96, stride=2, device='cuda:0', bias=False),
            nn.Conv2d(192 * 6, 192 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(192 * 6),
            nn.Conv2d(192 * 6, 160, kernel_size=1, padding=1, device='cuda:0', bias=False),

            # 7 bottleneck layer
            nn.Conv2d(160, 160 * 6, kernel_size=1, padding=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(160 * 6),
            nn.Conv2d(160 * 6, 320 * 6, kernel_size=3, padding=1, groups=160, stride=1, device='cuda:0', bias=False),
            nn.Conv2d(320 * 6, 320 * 6, kernel_size=1, device='cuda:0', bias=False),
            nn.ReLU6(inplace=True),
            nn.BatchNorm2d(320 * 6),
            nn.Conv2d(320 * 6, 320, kernel_size=1, padding=1, device='cuda:0', bias=False),

            # conv2d layer
            nn.Conv2d(320, 1280, kernel_size=1, padding=1, stride=1, device='cuda:0', bias=False),

            # avgpool layer
            nn.AvgPool2d(7, stride=1),

            # conv2d layer
            nn.Conv2d(1280, 43, kernel_size=1, padding=1, stride=1, device='cuda:0', bias=False),

            nn.Flatten(),
            nn.Linear(43 * 10 * 10, 43, device='cuda:0')

        )

    def forward(self, xb):
        return self.network(xb)


model = SimpsonsCnnModel()

lr = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)

num_epochs = 400
device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
to_device(model, device)


if __name__ == "__main__":
    model.train_epochs(train_dl, valid_dl)

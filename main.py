from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils import show_batch, show_img, get_default_device, to_device, get_log_num
from utils import DeviceDataLoader, evaluate, fit, plot_accuracies, plot_losses
import torchmetrics
import time
import numpy as np

# load images to tensors
crop = 192

path_to_imgs = "~\\Documents\\FLEET\\simpsons_dataset"
transform = transforms.Compose([transforms.Resize([crop, crop]), transforms.ToTensor(),
                                transforms.Normalize(std=0.5, mean=0.5)])
dataset = datasets.ImageFolder(path_to_imgs, transform=transform)

# Split data into train and valid parts
valid_size = 3000
train_size = len(dataset) - valid_size

train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# Create Batch data loader
batch_size = 64
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size, shuffle=True)


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

    def fit(self, train_dl, valid_dl, epochs=100, optimizer=torch.optim.SGD, lr=5e-5, num_log=get_log_num("Logs\\batches\\")):
        log = []
        self.train()
        optimizer = optimizer(self.parameters(), lr, momentum=0.9, weight_decay=1e-4)
        for epoch in range(epochs):
            train_losses = []
            accuracies = []
            start_time = time.time()
            # batch_cnt = 0
            for batch in train_dl:
                optimizer.zero_grad()
                target, loss = self.predict(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad()
                acc = self.accuracy(target, batch[1])
                print("Epoch: [{}], Batch accuracy: {:.4f}".format(epoch, acc.item()),
                      file=open("Logs\\batches\\log_{}.txt".format(num_log), 'a'))
                accuracies.append(acc.to('cpu'))
                # show_batch(batch, f"batch_{batch_cnt}")
                # batch_cnt += 1
            end_time = time.time()
            accuracies = np.array(accuracies)
            valid_eval = self.evaluate(valid_dl)
            result = {'epoch': epoch,
                      'train_loss': loss.item(),
                      'accuracy': accuracies.mean(),
                      'time': end_time - start_time,
                      'valid_loss': valid_eval[0],
                      'valid_accuracy': valid_eval[1]}
            print("| Epoch: [{}] | Train_loss: {:.4f} | Valid_loss: {:.4f} | Accuracy: {:.4f} | Valid_accuracy: {"
                  ":.4f} | Time: {:.4f} |".format(
                   result['epoch'],
                   result['train_loss'],
                   result['valid_loss'],
                   result['accuracy'],
                   result['valid_accuracy'],
                   result['time']), file=open("Logs\\epochs\\log_{}.txt".format(num_log), 'a'))
            log.append(result)
        return log


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, t=6, stride=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * t, kernel_size=3, padding=1, groups=nin, stride=stride)
        self.pointwise = nn.Conv2d(nin * t, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SimpsonsCnnModel(ImageClassification):
    def __init__(self):
        super(SimpsonsCnnModel, self).__init__()
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, padding=1, stride=1),
            nn.ReLU6(),
            nn.BatchNorm2d(32),
            depthwise_separable_conv(16, 64, t=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, kernel_size=1, padding=1, stride=1)
        )
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, padding=1, stride=1),
            nn.ReLU6(),
            nn.BatchNorm2d(16),
            depthwise_separable_conv(16, 64, t=6, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, kernel_size=1, padding=1, stride=1)
        )
        self.network = nn.Sequential(
            nn.BatchNorm2d(3),

            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(48*48*32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 43)
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
    res = model.fit(train_dl, valid_dl)
    print(res)
    # pass

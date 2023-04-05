from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torchvision.models import ResNet50_Weights
# from torch.utils.tensorboard import SummaryWriter
from utils import get_default_device, to_device
from utils import DeviceDataLoader
from tqdm import tqdm
from multiprocessing import Lock
import matplotlib.pyplot as plt

lock = Lock()
# load images to tensors
size = 224
crop = 224

path_to_imgs = "~\\Documents\\FLEET\\simpsons_dataset"
transform = transforms.Compose([transforms.Resize([size, size]),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation((-45, 45)),
                                transforms.ToTensor(),
                                transforms.Normalize(std=[0.229, 0.224, 0.225],
                                                     mean=[0.485, 0.456, 0.406]),

                                ])
dataset = datasets.ImageFolder(path_to_imgs, transform=transform)

# Split data into train and valid parts
valid_size = 3000
train_size = len(dataset) - valid_size

train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# Create Batch data loader
batch_size = 18
train_dl = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
valid_dl = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=4)


def train_step(model, train_dal, opt, pbar):
    model.train()
    running_loss = 0.
    for images, labels in train_dal:
        opt.zero_grad()

        output = model(images)
        loss = nn.CrossEntropyLoss()
        loss = loss(output, labels)
        loss.backward()
        opt.step()
        running_loss += loss
        with torch.no_grad():
            train_loss = (running_loss / train_size)
        pbar.set_description(f'Train loss for batch: {train_loss:.4f}')
        pbar.update(1)

    return train_loss.item()


def valid_step(model, valid_datal, pbar_2):
    model.eval()
    correct_total = 0.
    running_loss = 0.
    with torch.no_grad():
        for images, labels in valid_datal:
            output = model(images)
            prediction = output.argmax(dim=1)
            correct_total += prediction.eq(labels.view_as(prediction)).sum()

            loss = nn.CrossEntropyLoss()
            loss = loss(output, labels)

            running_loss += loss
            pbar_2.set_description(f'Running loss: {running_loss:.4f}')
            pbar_2.update(1)
    valid_loss = (running_loss / valid_size)
    accuracy = (correct_total / valid_size)
    return valid_loss.item(), accuracy.item()


def train_epochs(model, train_dl, valid_dl, optimizer, epochs=40, ):
    model.train()
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    pbar = tqdm(range(len(train_dl)), desc='Train loss for batch: ', leave=True)
    pbar.set_lock(lock)

    pbar_2 = tqdm(range(len(valid_dl)), desc='Running loss: ', leave=True)
    pbar_2.set_lock(lock)
    for epo in (pbar_1 := tqdm(range(epochs), leave=True, desc='Avg. train/loss: ')):
        train_loss = train_step(model, train_dl, optimizer, pbar)
        pbar.reset()

        valid_loss, valid_acc = valid_step(model, valid_dl, pbar_2)
        torch.save(model, f"{epo}.pt")
        pbar_2.reset()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        pbar_1.set_description(f'Metric: {valid_acc:.4f}. Avg. train/valid loss: {train_loss:.4f}/{valid_loss:.4f}')
    pbar_2.close()
    pbar.close()
    pbar_1.close()
    figure = plt.figure(figsize=(16, 12))
    plt.plot(train_losses[1:], label='train loss', )
    plt.plot(valid_losses[1:], label='valid loss')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    figure.show()
    plt.plot(valid_accuracies)
    plt.show()


res_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.IMAGENET1K_V1, verbose=False)

num_epochs = 5
device = get_default_device()

res_net.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 43)
).to(device)

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
to_device(res_net, device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(res_net.parameters(), lr=5e-3, weight_decay=5e-4)

if __name__ == '__main__':

    test_model = torch.load("0.pt")
    print(test_model.fc[0].weight)
    _check = torch.load("1.pt")
    print(_check.fc[0].weight)
    # train_epochs(res_net, train_dl, valid_dl, optim, num_epochs)


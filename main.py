from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torchvision.models import ResNet50_Weights
import torch.optim.lr_scheduler as sched
from utils import get_default_device, to_device, map_init
from utils import DeviceDataLoader, get_log_num
from tqdm import tqdm
import math
from multiprocessing import Lock
import matplotlib.pyplot as plt

lock = Lock()
# load images to tensors
size = 156
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
batch_size = 100
train_dl = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=4)
valid_dl = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=4)


def train_step(model, train_dal, opt, bar, metric, epo):
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
        bar.set_description(f'[{epo}] Metric: {metric:.4f}. Training: {train_loss:.4f} loss')
        bar.update(batch_size)

    return train_loss.item()


def valid_step(model, valid_datal, bar, metric, tr_l, epo):
    model.eval()
    correct_total = 0.
    running_loss = 0.

    preds = []
    cors = []

    with torch.no_grad():
        for images, labels in valid_datal:
            output = model(images)
            prediction = torch.argmax(model(images), dim=1)
            preds.append(prediction)
            correct_total += sum(prediction == labels).item()
            cors.append(correct_total)
            loss = nn.CrossEntropyLoss()
            loss = loss(output, labels)

            running_loss += loss
            bar.set_description(
                f'[{epo}] Metric: {metric:.4f}. Average train loss: {tr_l:.4f}. Validation: {running_loss:.4f}'
                f' loss')
            bar.update(batch_size)

    valid_loss = (running_loss / valid_size)
    accuracy = (correct_total / valid_size)
    return valid_loss.item(), accuracy


def train_epochs(model, train_dl, valid_dl, optimizer, epochs=40):
    model.train()
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    lrs = [0.0]

    metr = 0.
    m_bar = tqdm(range(valid_size + train_size), desc=f"Metric: {metr:.4f}", leave=True)
    m_bar.set_lock(lock)

    scheduler = sched.LambdaLR(optimizer, math.sin)

    for epo in range(epochs):
        train_loss = train_step(model, train_dl, optimizer, m_bar, metr, epo)

        valid_loss, valid_acc = valid_step(model, valid_dl, m_bar, metr, train_loss, epo)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        best_checkpoint = get_log_num("checkpoints/")
        if valid_acc >= valid_accuracies[-1] and len(valid_accuracies) > 1 and valid_acc > best_checkpoint:
            torch.save(model, f"checkpoints/best_{valid_acc:.2f}.pt")
        else:
            torch.save(model, f"checkpoints/last_.pt")
        metr = valid_acc

        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        after = optimizer.param_groups[0]['lr']

        lrs.append(after)
        m_bar.set_description(f'[{epo}] Metric: {metr:.4f}. [{before_lr:.5f}/{after:.5f}] Avg. train/valid loss: '
                              f'{train_loss:.4f}/{valid_loss:.4f}')
        m_bar.reset()
    m_bar.close()
    figure = plt.figure(figsize=(16, 12))
    plt.plot(train_losses[1:], label='train loss', )
    plt.plot(valid_losses[1:], label='valid loss')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    figure.show()
    # plt.plot(lrs)
    # plt.show()
    plt.plot(valid_accuracies)
    plt.show()


res_net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.IMAGENET1K_V1, verbose=False)

num_epochs = 4
device = get_default_device()

res_net.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(128),
    nn.Linear(128, 42)
).to(device)

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
to_device(res_net, device)

criterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    # best = get_log_num("checkpoints/")
    model = torch.load(f"checkpoints/backup/0.95_156x156_100bs.pt")
    path_to_test = "kaggle_simpson_testset"
    trans = transforms.Compose(
        [transforms.Resize([156, 156]),
        transforms.ToTensor()]
    )
    image = datasets.ImageFolder(path_to_test, transform=trans)
    pred = model(image)
    mapping = map_init("simpsons_dataset")
    print(mapping[pred.argmax(dim=1).item()])
    # optim = torch.optim.Adam(res_net.parameters(), lr=1e-3, weight_decay=5e-4)
    # train_epochs(res_net, train_dl, valid_dl, optim, num_epochs)

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import time
import re
import os
import numpy as np


def show_img(dataset, idx):
    """Shows image from dataset"""
    img, label = dataset[idx]
    print("Label: ", dataset.classes[label], '(' + str(label) + ')')
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.show()


def show_batch(batch, name):
    """Shows batch from dataloader"""
    imgs, labels = batch
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(imgs.to('cpu'), nrow=16).permute(1, 2, 0))
    plt.savefig("batch_imgs\\"+name)
    plt.close()


def get_log_num(path):

    log_list = list(filter(re.compile('log_*.').match, os.listdir(path)))
    log_list = list([s.replace('log_', '') for s in log_list])
    log_list = list([s.replace('.txt', '') for s in log_list])
    if not log_list:
        return 0
    return max(list(map(int, log_list)))+1


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, model, train_loader, optimizer):
    history = []
    model.train()
    for epoch in range(epochs):
        # Training Phase
        train_losses = []
        accuracies = []
        start_time = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            tar, loss = model.predict(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            print("Epoch: [{}], Batch accuracy: {:.4f}".format(epoch, model.accuracy(tar, batch[1]).item()),
                  file=open("log.txt", 'w'))
            accuracies.append(model.accuracy(tar, batch[1]).to('cpu'))
        # Validation phase
        end_time = time.time()
        accuracies = np.array(accuracies)
        result = {'epoch': epoch, 'train_loss': loss.item(), 'accuracy': accuracies.mean(),
                  'time': end_time - start_time}
        print("| Epoch: [{}] | Train_loss: {:.4f} | Accuracy: {:.4f} | Time: {:.4f} |".format(result['epoch'],
                                                                                              result['train_loss'],
                                                                                              result['accuracy'],
                                                                                              result['time']))
        history.append(result)
    return history


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

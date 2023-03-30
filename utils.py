import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import re
import os


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

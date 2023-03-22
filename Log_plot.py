import matplotlib.pyplot as plt
from utils import get_log_num


def plot_batch_acc():
    with open('Logs\\batches\\log_{}.txt'.format(get_log_num("Logs\\batches\\")-1), 'r') as f:
        log_dic = {}
        for line in f:
            epoch = line[8:line.find(']')]
            if epoch not in log_dic:
                log_dic[epoch] = [line[-7:].replace('\n', '')[:-2]]
            else:
                log_dic[epoch].append(line[-7:].replace('\n', '')[:-2])
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['figure.autolayout'] = True
    if len(log_dic) != 1:
        for epoch in log_dic:
            idx = int(epoch)
            log_dic[epoch] = list(map(float, log_dic[epoch]))
            plt.subplot((len(log_dic)) // 10 + 1, len(log_dic) % 10 if len(log_dic) < 10 else 10, (idx + 1))
            plt.plot(log_dic[epoch])
            plt.ylim(bottom=0, top=1)

    else:
        plt.plot(log_dic['0'])
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')

    plt.show()


def plot_epochs_log():
    plt.rcParams['figure.figsize'] = [6, 6]
    plt.rcParams['figure.autolayout'] = True
    with open('Logs\\epochs\\log_{}.txt'.format(get_log_num("Logs\\epochs\\")-1), 'r') as f:
        t_loss = []
        v_loss = []
        t_acc = []
        v_acc = []
        for line in f:
            tl_i = line.find('Train_loss: ')
            vl_i = line.find('Valid_loss: ')
            ta_i = line.find('Accuracy: ')
            va_i = line.find('Valid_accuracy: ')
            t_loss.append(line[tl_i + 12: tl_i + 18])
            v_loss.append(line[vl_i + 12: vl_i + 18])
            t_acc.append(line[ta_i + 10: ta_i + 16])
            v_acc.append(line[va_i + 16: va_i + 22])
        fig, ax = plt.subplots(2, 2, sharex='all', sharey='row')
        ax[0, 0].plot(list(map(float, t_loss)))
        ax[0, 0].set_title("Train loss")
        ax[0, 1].plot(list(map(float, v_loss)))
        ax[0, 1].set_title("Valid loss")
        ax[1, 0].plot(list(map(float, t_acc)))
        ax[1, 0].set_title("Train accuracy")
        ax[1, 1].plot(list(map(float, v_acc)))
        ax[1, 1].set_title("Valid accuracy")
    plt.show()


if __name__ == "__main__":
    plot_epochs_log()
    plot_batch_acc()

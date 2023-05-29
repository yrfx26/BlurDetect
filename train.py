import os.path

import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from matplotlib import pyplot as plt

from model import model


train_dataset_dir = "dataset/CERTH_ImageBlurDataset/Train"
val_dataset_dir = "dataset/CERTH_ImageBlurDataset/Val"
batch_size = 32
lr = 0.001
num_epochs = 1
num_workers = 0


def get_save_dir():
    num = len(os.listdir("runs")) + 1
    return os.path.join("runs", f"{num:03}")


def write_logs(logs, save_path):
    with open(save_path, "w") as f:
        for i, log in enumerate(logs):
            s = f"epoch:{i}\ttrain_loss:{log[0][0]}\ttrain_acc:{log[0][1]}\t" \
                f"val_loss:{log[1][0]}\tval_acc:{log[1][1]}\n"
            f.write(s)


def draw_logs(logs_txt, save_dir):
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    with open(logs_txt, 'r') as f:
        for line in f.readlines():
            l = line.strip().split("\t")
            train_loss = float(l[1].split(":")[1])
            train_acc = float(l[2].split(":")[1])
            test_loss = float(l[3].split(":")[1])
            test_acc = float(l[4].split(":")[1])
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

    x = list(range(len(train_acc_list)))
    plt.plot(x, train_acc_list, label="train_acc")
    plt.plot(x, test_acc_list, label="test_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "acc.png"))

    x = list(range(len(train_acc_list)))
    plt.plot(train_loss_list, label="train_loss")
    # plt.plot(test_loss_list, label="test_loss")
    # plt.ylim([0.005, 0.015])
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.png"))


def accuracy(y_hat, y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def train_one_epoch(model, train_iter, opt, loss, device):
    sum_num = 0
    sum_correct = 0
    sum_loss = 0
    print("开始训练------------------------------------------")
    model.train()
    for batch_id, (x, y) in enumerate(train_iter):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        y_hat = model(x)
        l = loss(y_hat, y)
        l.backward()
        opt.step()

        with torch.no_grad():
            num_loss = float(l.sum())
            num_correct = accuracy(y_hat, y)
            num = y.numel()
            sum_loss += num_loss
            sum_correct += num_correct
            sum_num += num
            if batch_id % 10 == 0 and batch_id != 0:
                print(f"batch:{batch_id}"
                      f"\tloss:{num_loss / num:.5}\ttrain_acc:{num_correct / num:.5}")

    return sum_loss/sum_num, sum_correct/sum_num


def test_once(model, valid_iter, loss, device):
    sum_num = 0
    sum_correct = 0
    sum_loss = 0
    print("开始验证------------------------------------------")
    model.eval()
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(valid_iter):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            l = loss(y_hat, y)

            sum_loss += float(l.sum())
            sum_correct += accuracy(y_hat, y)
            sum_num += y.numel()
            if batch_id % 10 == 0 and batch_id != 0:
                print(f"batch:{batch_id}"
                      f"\ttest loss:{sum_loss / sum_num:.5}\ttest_acc:{sum_correct / sum_num:.5}")

    return sum_loss / sum_num, sum_correct / sum_num


def train(model, train_iter, opt, num_epochs, valid_iter=None, device=torch.device('cuda'), is_save=True):
    model.to(device)
    loss = nn.CrossEntropyLoss()
    logs = []
    for epoch in range(num_epochs):
        log = []
        print(f"\nepoch: {epoch}")
        log.append(train_one_epoch(model=model, train_iter=train_iter, opt=opt, loss=loss, device=device))
        log.append(test_once(model=model, valid_iter=valid_iter, loss=loss, device=device))

        print(f"train loss: {log[0][0]:.5f}, train acc: {log[0][1]:.5f}\n"
              f"val loss: {log[1][0]:.5f}, val acc: {log[1][1]:.5f}")
        logs.append(log)

    if is_save:
        save_dir = get_save_dir()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logs_path = os.path.join(save_dir, "logs.txt")

        write_logs(logs, save_path=logs_path)
        draw_logs(logs_path, save_dir)
        torch.save(model, os.path.join(save_dir, "model.pth"))

    return logs


if __name__ == '__main__':
    # 获取数据
    # label_txt = r"F:\0\projects\PythonProjects\BlurDetection\dataset\label.txt"
    # class2num_txt = r"F:\0\projects\PythonProjects\BlurDetection\dataset\class2num"
    # train_dataset = TrainDataSet(label_txt, class2num_txt)

    # 数据增强
    train_transform = T.Compose([
        # T.RandomRotation(180),
        # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(448),
        T.ToTensor(),
    ])
    valid_transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
    ])

    # data_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_dir, transform=train_transform)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valid_dataset = torchvision.datasets.ImageFolder(root=val_dataset_dir, transform=valid_transform)
    valid_iter = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # train_iter = get_data_iter(
    #     root=r"dataset\CERTH_ImageBlurDataset\Train",
    #     batch_size=batch_size,
    #     transforms=train_transforms,)
    # val_iter = get_data_iter(
    #     root=r"dataset\CERTH_ImageBlurDataset\Val",
    #     batch_size=batch_size,
    #     transforms=valid_transforms)

    # 定义模型
    my_model = model.Model()
    # 设置训练参数

    opt = optim.Adam(my_model.parameters(), lr=lr)
    num_epochs = 1
    logs = train(my_model, train_iter, opt, num_epochs, valid_iter)
    # torch.save(my_model, 'models/model_1.pth')

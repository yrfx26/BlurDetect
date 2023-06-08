from matplotlib import pyplot as plt
import os


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
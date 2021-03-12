#!/usr/bin/env python3

import torch
import torchvision
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import CNNModel
from dataset import Dataset
from prioritized_dataset import PrioritizedDataset, PrioritizedDataloader

# define constants
image_size = 32
image_channel = 3
class_num = 10


def calc_loss(model, data_loader, device):
    with torch.no_grad():
        loss = 0
        accuracy = 0
        data_num = 0
        accuracy_for_each_class = [0 for _ in range(class_num)]
        data_num_for_each_class = [0 for _ in range(class_num)]
        model.eval()
        for minibatch in data_loader:
            x, y = minibatch
            x, y = x.to(device), y.to(device)
            classify = model.forward(x)
            curr_loss_ce = torch.nn.functional.cross_entropy(classify, y)
            loss += curr_loss_ce.item() * x.shape[0]
            _, predicted = torch.max(classify, 1)
            accuracy += (predicted == y).sum().item()

            for i in range(x.shape[0]):
                accuracy_for_each_class[y[i]] += (predicted[i] == y[i]).item()
                data_num_for_each_class[y[i]] += 1

        data_num = sum(data_num_for_each_class)
        loss /= data_num
        accuracy /= data_num
        for i in range(class_num):
            accuracy_for_each_class[i] /= data_num_for_each_class[i]
    return loss, accuracy, accuracy_for_each_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--data_num_of_imbalanced_class", type=int, default=5000)
    parser.add_argument("--copy_imbalanced_class", action="store_true")
    parser.add_argument("--use_prioritized_dataset", action="store_true")
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=1.0)
    args = parser.parse_args()

    # prepare data_loader
    transform_augment = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomCrop(32, padding=4)])
    transform_normal = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_dir = "../data"
    trainset = Dataset(
        root=data_dir,
        transform=transform_augment,
        data_num_of_imbalanced_class=args.data_num_of_imbalanced_class,
        copy_imbalanced_class=args.copy_imbalanced_class)
    train_size = int(len(trainset) * 0.9)
    valid_size = len(trainset) - train_size
    trainset, validset = torch.utils.data.random_split(trainset, [train_size, valid_size])
    validset.transform = transform_normal
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_normal)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.use_prioritized_dataset:
        trainset = PrioritizedDataset(trainset)
        trainloader = PrioritizedDataloader(trainset, batch_size=args.batch_size)

    # create model
    model = CNNModel(image_size, image_channel)
    if args.saved_model_path is not None:
        model.load_state_dict(torch.load(args.saved_model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    save_log_dir = "../result"
    save_model_path = f"{save_log_dir}/best_model.pt"

    # optimizer
    optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epoch)

    # log
    valid_df = pd.DataFrame(columns=["time(seconds)", "epoch", "loss", "accuracy", "mean_accuracy"] +
                            [f"accuracy_of_class{i}" for i in range(class_num)])
    start = time.time()
    best_accuracy = -float("inf")

    # training step
    for epoch in range(args.epoch):
        # train
        model.train()
        for step, minibatch in enumerate(trainloader):
            x, y = minibatch
            x, y = x.to(device), y.to(device)

            if args.use_mixup:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha) if args.mixup_alpha > 0 else 1
                x_r = torch.roll(x, 1, 0)

                # case(1) mix input
                mixed_x = lam * x + (1 - lam) * x_r
                classify = model.forward(mixed_x)

                # case(2) mix representation
                # representation = model.encode(x)
                # mixed_representation = lam * representation + (1 - lam) * torch.roll(representation, 1, 0)
                # classify = model.classifier(mixed_representation)

                # mix loss
                y_r = torch.roll(y, 1, 0)
                loss1 = torch.nn.functional.cross_entropy(classify, y, reduction="none")
                loss2 = torch.nn.functional.cross_entropy(classify, y_r, reduction="none")
                loss = lam * loss1 + (1 - lam) * loss2
            else:
                classify = model.forward(x)
                loss = torch.nn.functional.cross_entropy(classify, y, reduction="none")

            if args.use_prioritized_dataset:
                trainset.update(loss)

            loss = loss.mean()

            _, predicted = torch.max(classify, 1)
            accuracy = (predicted == y).sum().item() / x.shape[0]
            elapsed = time.time() - start
            loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{step + 1}\t{loss:.4f}\t{accuracy * 100:.1f}"
            print(loss_str, end="\r")

            optim.zero_grad()
            loss.backward()
            optim.step()

        # validation
        valid_loss, valid_accuracy, valid_accuracy_for_each_class = calc_loss(model, validloader, device)
        valid_mean_accuracy = np.mean(valid_accuracy_for_each_class)
        elapsed = time.time() - start
        s = pd.Series([elapsed, int(epoch + 1), valid_loss,
                       valid_accuracy, valid_mean_accuracy] + valid_accuracy_for_each_class, index=valid_df.columns)
        valid_df = valid_df.append(s, ignore_index=True)
        loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{valid_loss:.4f}\t{valid_accuracy * 100:.1f}\t{valid_mean_accuracy * 100:.1f}"
        print(" " * 100, end="\r")
        print(loss_str)

        if valid_mean_accuracy > best_accuracy:
            best_accuracy = valid_mean_accuracy
            torch.save(model.state_dict(), save_model_path)

        scheduler.step()

    # load best model
    model.load_state_dict(torch.load(save_model_path))

    # save validation loss
    valid_df.to_csv(f"{save_log_dir}/valid_loss.tsv", sep="\t")

    # plot validation loss
    valid_df.plot(x="epoch", y=["loss", "accuracy"], subplots=True, layout=(2, 1), marker=".", figsize=(16, 9))
    plt.savefig(f"{save_log_dir}/valid_loss.png", bbox_inches="tight", pad_inches=0.05)
    plt.clf()
    valid_df.plot(x="epoch", y=[f"accuracy_of_class{i}" for i in range(class_num)], marker=".", figsize=(16, 9))
    plt.savefig(f"{save_log_dir}/valid_accuracy_for_each_class.png", bbox_inches="tight", pad_inches=0.05)

    # save test loss
    test_loss, test_accuracy, test_accuracy_for_each_class = calc_loss(model, testloader, device)
    test_df = pd.DataFrame(data=[[test_loss, test_accuracy] + test_accuracy_for_each_class],
                           columns=["loss", "accuracy"] + [f"accuracy_of_class{i}" for i in range(class_num)])
    test_df.to_csv(f"{save_log_dir}/test_loss.tsv", sep="\t")


if __name__ == "__main__":
    main()

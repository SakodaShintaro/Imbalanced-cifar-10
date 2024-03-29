#!/usr/bin/env python3

import torch
import torchvision
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from model import CNNModel
from dataset import Dataset
from prioritized_dataset import PrioritizedDataset, PrioritizedDataloader

# define constants
image_size = 32
image_channel = 3
class_num = 10


def calc_loss(model, data_loader, device, args):
    with torch.no_grad():
        loss_mse = 0
        loss_ce = 0
        loss_sum = 0
        accuracy = 0
        data_num = 0
        accuracy_for_each_class = [0 for _ in range(class_num)]
        data_num_for_each_class = [0 for _ in range(class_num)]
        model.eval()
        for minibatch in data_loader:
            x, y = minibatch
            x, y = x.to(device), y.to(device)
            reconstruct, classify = model.forward(x)
            curr_loss_mse = torch.nn.functional.mse_loss(reconstruct, x)
            curr_loss_ce = torch.nn.functional.cross_entropy(classify, y)
            curr_loss_sum = args.coefficient_of_mse * curr_loss_mse + args.coefficient_of_ce * curr_loss_ce
            loss_mse += curr_loss_mse.item() * x.shape[0]
            loss_ce += curr_loss_ce.item() * x.shape[0]
            loss_sum += curr_loss_sum.item() * x.shape[0]
            _, predicted = torch.max(classify, 1)
            accuracy += (predicted == y).sum().item()

            for i in range(x.shape[0]):
                accuracy_for_each_class[y[i]] += (predicted[i] == y[i]).item()
                data_num_for_each_class[y[i]] += 1

        data_num = sum(data_num_for_each_class)
        loss_mse /= data_num
        loss_ce /= data_num
        loss_sum /= data_num
        accuracy /= data_num
        for i in range(class_num):
            accuracy_for_each_class[i] /= data_num_for_each_class[i]
    return loss_sum, loss_mse, loss_ce, accuracy, accuracy_for_each_class


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--data_num_of_imbalanced_class", type=int, default=2500)
    parser.add_argument("--coefficient_of_mse", type=float, default=1)
    parser.add_argument("--coefficient_of_ce", type=float, default=1)
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
    model = CNNModel(image_size, image_channel, args.hidden_size, class_num)
    if args.saved_model_path is not None:
        model.load_state_dict(torch.load(args.saved_model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer
    optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epoch)

    # log
    valid_df = pd.DataFrame(columns=['time(seconds)', 'epoch', 'sum', 'reconstruct_mse', 'cross_entropy',
                                     'accuracy', "mean_accuracy"] + [f'accuracy_of_class{i}' for i in range(class_num)])
    start = time.time()
    best_accuracy = -float("inf")

    # training step
    for epoch in range(args.epoch):
        # train
        model.train()
        for step, minibatch in enumerate(trainloader):
            x, y = minibatch
            if args.use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=args.mixup_alpha)
                y_a, y_b = y_a.to(device), y_b.to(device)
            x, y = x.to(device), y.to(device)
            reconstruct, classify = model.forward(x)
            loss_mse = torch.nn.functional.mse_loss(reconstruct, x, reduction="none").mean([1, 2, 3])

            if args.use_mixup:
                loss_ce1 = torch.nn.functional.cross_entropy(classify, y_a, reduction="none")
                loss_ce2 = torch.nn.functional.cross_entropy(classify, y_b, reduction="none")
                loss_ce = lam * loss_ce1 + (1 - lam) * loss_ce2
            else:
                loss_ce = torch.nn.functional.cross_entropy(classify, y, reduction="none")

            loss_sum = args.coefficient_of_mse * loss_mse + args.coefficient_of_ce * loss_ce

            if args.use_prioritized_dataset:
                trainset.update(loss_sum)

            loss_mse = loss_mse.mean()
            loss_ce = loss_ce.mean()
            loss_sum = loss_sum.mean()

            _, predicted = torch.max(classify, 1)
            accuracy = (predicted == y).sum().item() / x.shape[0]
            elapsed = time.time() - start
            loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{step + 1}\t{loss_sum:.4f}\t{loss_mse:.4f}\t{loss_ce:.4f}\t{accuracy * 100:.1f}"
            print(loss_str, end="\r")

            optim.zero_grad()
            loss_sum.backward()
            optim.step()

        # validation
        valid_loss_sum, valid_loss_mse, valid_loss_ce, valid_accuracy, valid_accuracy_for_each_class = calc_loss(model, validloader, device, args)
        valid_mean_accuracy = np.mean(valid_accuracy_for_each_class)
        elapsed = time.time() - start
        s = pd.Series([elapsed, int(epoch + 1), valid_loss_sum, valid_loss_mse, valid_loss_ce,
                       valid_accuracy, valid_mean_accuracy] + valid_accuracy_for_each_class, index=valid_df.columns)
        valid_df = valid_df.append(s, ignore_index=True)
        loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{valid_loss_sum:.4f}\t{valid_loss_mse:.4f}\t{valid_loss_ce:.4f}\t{valid_accuracy * 100:.1f}\t{valid_mean_accuracy * 100:.1f}"
        print(" " * 100, end="\r")
        print(loss_str)

        if valid_mean_accuracy > best_accuracy:
            best_accuracy = valid_mean_accuracy
            torch.save(model.state_dict(), "../result/model/model.pt")

        scheduler.step()

    # load best model
    model.load_state_dict(torch.load("../result/model/model.pt"))

    # save validation loss
    valid_df.to_csv("../result/loss_log/validation_loss.tsv", sep="\t")

    # plot validation loss
    valid_df.plot(x="epoch", y=['sum', 'reconstruct_mse', 'cross_entropy', 'accuracy'], subplots=True, layout=(2, 2), marker=".", figsize=(16, 9))
    plt.savefig('../result/loss_log/validation_loss.png', bbox_inches="tight", pad_inches=0.05)
    plt.clf()
    valid_df.plot(x="epoch", y=[f'accuracy_of_class{i}' for i in range(class_num)], marker=".", figsize=(16, 9))
    plt.savefig('../result/loss_log/accuracy_for_each_class.png', bbox_inches="tight", pad_inches=0.05)

    # save test loss
    with open("../result/loss_log/test_loss.txt", "w") as f:
        test_loss_sum, test_loss_mse, test_loss_ce, test_accuracy, test_accuracy_for_each_class = calc_loss(model, testloader, device, args)
        f.write("loss_sum\tloss_mse\tloss_ce\taccuracy")
        for i in range(class_num):
            f.write(f"\taccuracy_of_class{i}")
        f.write("\n")

        f.write(f"{test_loss_sum:.4f}\t{test_loss_mse:.4f}\t{test_loss_ce:.4f}\t{test_accuracy * 100:.1f}")
        for i in range(class_num):
            f.write(f"\t{test_accuracy_for_each_class[i]}")
        f.write("\n")

    # show reconstruction
    result_image_dir = "../result/image/"
    with torch.no_grad():
        model.eval()
        for minibatch in testloader:
            x, y = minibatch
            x, y = x.to(device), y.to(device)
            out, _ = model.forward(x)

            x = (x + 1) / 2 * 256
            x = x.to(torch.uint8)

            out = (out + 1) / 2 * 256
            out = out.to(torch.uint8)

            for i in range(args.batch_size):
                origin = x[i].reshape([image_channel, image_size, image_size])
                origin = origin.permute([1, 2, 0])
                origin = origin.cpu().numpy()

                pred = out[i].reshape([image_channel, image_size, image_size])
                pred = pred.permute([1, 2, 0])
                pred = pred.cpu().numpy()

                pil_img0 = Image.fromarray(origin)
                pil_img0.save(f"{result_image_dir}/{i}-0.png")
                pil_img1 = Image.fromarray(pred)
                pil_img1.save(f"{result_image_dir}/{i}-1.png")
            exit()


if __name__ == "__main__":
    main()

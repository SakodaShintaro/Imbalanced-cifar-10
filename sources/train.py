#!/usr/bin/env python3

import torch
import torchvision
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from model import LinearModel, CNNModel
from dataset import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--data_num_of_imbalanced_class", type=int, default=2500)
    parser.add_argument("--coefficient_of_mse", type=float, default=1)
    parser.add_argument("--coefficient_of_ce", type=float, default=1)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    root_dir = "../data"

    trainset = Dataset(root=root_dir, transform=transform, data_num_of_imbalanced_class=args.data_num_of_imbalanced_class)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    image_size = 32
    image_channel = 3
    class_num = 10

    model = CNNModel(image_size, image_channel, args.hidden_size, class_num)
    if args.saved_model_path is not None:
        model.load_state_dict(torch.load(args.saved_model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device)

    optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    shceduler = torch.optim.lr_scheduler.MultiStepLR(optim, [args.epoch // 2, args.epoch * 3 // 4], gamma=0.1)

    valid_df = pd.DataFrame(columns=['time(seconds)', 'epoch', 'sum', 'reconstruct_mse', 'cross_entropy', 'accuracy'])

    start = time.time()
    for epoch in range(args.epoch):
        # train
        model.train()
        for step, minibatch in enumerate(trainloader):
            x, y = minibatch
            x, y = x.to(device), y.to(device)
            reconstruct, classify = model.forward(x)
            loss_mse = torch.nn.functional.mse_loss(reconstruct, x)
            loss_ce = torch.nn.functional.cross_entropy(classify, y)
            loss_sum = args.coefficient_of_mse * loss_mse + args.coefficient_of_ce * loss_ce
            _, predicted = torch.max(classify, 1)
            accuracy = (predicted == y).sum().item() / x.shape[0]
            elapsed = time.time() - start
            loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{step + 1}\t{loss_sum:.4f}\t{loss_mse:.4f}\t{loss_ce:.4f}\t{accuracy * 100:.1f}"
            print(loss_str, end="\r")

            optim.zero_grad()
            loss_sum.backward()
            optim.step()

        # validation
        with torch.no_grad():
            validation_loss_mse = 0
            validation_loss_ce = 0
            validation_loss_sum = 0
            validation_accuracy = 0
            data_num = 0
            model.eval()
            for minibatch in testloader:
                x, y = minibatch
                x, y = x.to(device), y.to(device)
                reconstruct, classify = model.forward(x)
                loss_mse = torch.nn.functional.mse_loss(reconstruct, x)
                loss_ce = torch.nn.functional.cross_entropy(classify, y)
                loss_sum = args.coefficient_of_mse * loss_mse + args.coefficient_of_ce * loss_ce
                validation_loss_mse += loss_mse.item() * x.shape[0]
                validation_loss_ce += loss_ce.item() * x.shape[0]
                validation_loss_sum += loss_sum.item() * x.shape[0]
                _, predicted = torch.max(classify, 1)
                validation_accuracy += (predicted == y).sum().item()
                data_num += x.shape[0]
            validation_loss_mse /= data_num
            validation_loss_ce /= data_num
            validation_loss_sum /= data_num
            validation_accuracy /= data_num
        elapsed = time.time() - start
        s = pd.Series([elapsed, int(epoch + 1), validation_loss_sum, validation_loss_mse,
                       validation_loss_ce, validation_accuracy], index=valid_df.columns)
        valid_df = valid_df.append(s, ignore_index=True)
        loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{validation_loss_sum:.4f}\t{validation_loss_mse:.4f}\t{validation_loss_ce:.4f}\t{validation_accuracy * 100:.1f}           "
        print(loss_str)

        shceduler.step()

    # save model
    torch.save(model.state_dict(), "../result/model/model.pt")

    # save validation loss
    valid_df.to_csv("../result/loss_log/validation_loss.tsv", sep="\t")

    # plot validation loss
    valid_df.plot(x="epoch", y=['sum', 'reconstruct_mse', 'cross_entropy', 'accuracy'], subplots=True, layout=(2, 2), marker=".", figsize=(16, 9))
    plt.savefig('../result/loss_log/validation_loss.png', bbox_inches="tight", pad_inches=0.01)

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

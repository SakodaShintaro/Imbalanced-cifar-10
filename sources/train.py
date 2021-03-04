#!/usr/bin/env python3

import torch
import torchvision
import argparse
import time
from PIL import Image
from model import Model
from dataset import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--data_num_of_imbalanced_class", type=int, default=2500)
    parser.add_argument("--coefficient_of_mse", type=float, default=1)
    parser.add_argument("--coefficient_of_ce", type=float, default=1)
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

    model = Model(image_size * image_size * image_channel, args.hidden_size, class_num)
    if args.saved_model_path is not None:
        model.load_state_dict(torch.load(args.saved_model_path))

    optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    start = time.time()
    for epoch in range(10):
        # train
        model.train()
        for step, minibatch in enumerate(trainloader):
            x, y = minibatch
            reconstruct, classify = model.forward(x)
            loss_mse = torch.nn.functional.mse_loss(reconstruct, x)
            loss_ce = torch.nn.functional.cross_entropy(classify, y)
            _, predicted = torch.max(classify, 1)
            accuracy = (predicted == y).sum().item() / x.shape[0]
            elapsed = time.time() - start
            loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{step + 1}\t{loss_mse:.4f}\t{loss_ce:.4f}\t{accuracy * 100:.1f}"
            print(loss_str, end="\r")

            optim.zero_grad()
            (loss_mse + loss_ce).backward()
            optim.step()

        # validation
        with torch.no_grad():
            validation_loss_mse = 0
            validation_loss_ce = 0
            validation_accuracy = 0
            data_num = 0
            model.eval()
            for minibatch in testloader:
                x, y = minibatch
                reconstruct, classify = model.forward(x)
                loss_mse = torch.nn.functional.mse_loss(reconstruct, x)
                loss_ce = torch.nn.functional.cross_entropy(classify, y)
                validation_loss_mse += loss_mse * x.shape[0]
                validation_loss_ce += loss_ce * x.shape[0]
                _, predicted = torch.max(classify, 1)
                validation_accuracy += (predicted == y).sum().item()
                data_num += x.shape[0]
            validation_loss_mse /= data_num
            validation_loss_ce /= data_num
            validation_accuracy /= data_num
        elapsed = time.time() - start
        loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{validation_loss_mse:.4f}\t{validation_loss_ce:.4f}\t{validation_accuracy * 100:.1f}           "
        print(loss_str)

    # save model
    torch.save(model.state_dict(), "../result/model/model.pt")

    # show reconstruction
    result_image_dir = "../result/image/"
    with torch.no_grad():
        model.eval()
        for minibatch in testloader:
            x, y = minibatch
            out, _ = model.forward(x)

            x = (x + 1) / 2 * 256
            x = x.to(torch.uint8)

            out = (out + 1) / 2 * 256
            out = out.to(torch.uint8)

            for i in range(args.batch_size):
                origin = x[i].reshape([image_channel, image_size, image_size])
                origin = origin.permute([1, 2, 0])
                origin = origin.numpy()

                pred = out[i].reshape([image_channel, image_size, image_size])
                pred = pred.permute([1, 2, 0])
                pred = pred.numpy()

                pil_img0 = Image.fromarray(origin)
                pil_img0.save(f"{result_image_dir}/{i}-0.png")
                pil_img1 = Image.fromarray(pred)
                pil_img1.save(f"{result_image_dir}/{i}-1.png")
            exit()


if __name__ == "__main__":
    main()

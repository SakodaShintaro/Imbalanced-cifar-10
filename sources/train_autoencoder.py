#!/usr/bin/env python3

import torch
import torchvision
import argparse
import time
from PIL import Image
from autoencoder import AutoEncoder
from dataset import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    root_dir = "../data"

    trainset = Dataset(root=root_dir, transform=transform,
                       data_num_of_imbalanced_class=2500)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    image_size = 32
    image_channel = 3

    auto_encoder = AutoEncoder(
        image_size * image_size * image_channel, args.hidden_size)

    optim = torch.optim.SGD(auto_encoder.parameters(), lr=20.0)

    start = time.time()
    for epoch in range(10):
        # train
        auto_encoder.train()
        for step, minibatch in enumerate(trainloader):
            x, y = minibatch
            x = x.flatten(1)
            optim.zero_grad()
            out = auto_encoder.forward(x)
            loss = torch.nn.functional.mse_loss(out, x)
            elapsed = time.time() - start
            loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{step + 1}\t{loss:.4f}"
            print(loss_str, end="\r")
            loss.backward()
            optim.step()

        # validation
        with torch.no_grad():
            validation_loss = 0
            data_num = 0
            auto_encoder.eval()
            for minibatch in testloader:
                x, y = minibatch
                x = x.flatten(1)
                out = auto_encoder.forward(x)
                loss = torch.nn.functional.mse_loss(out, x, reduction="mean")
                validation_loss += loss * x.shape[0]
                data_num += x.shape[0]
            validation_loss /= data_num
        elapsed = time.time() - start
        loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{validation_loss:.4f}          "
        print(loss_str)

    # save model
    torch.save(auto_encoder, "../result/model/autoencoder.pt")

    # show reconstruction
    result_image_dir = "../result/image/"
    with torch.no_grad():
        auto_encoder.eval()
        for minibatch in testloader:
            x, y = minibatch
            x = x.flatten(1)
            out = auto_encoder.forward(x)

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

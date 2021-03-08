This repository contains code for deep neural network training experiments when there is little data in the "bird", "deer", and "truck" classes for cifar-10. I aim to deal with data imbalance by introducing a reconstruction mechanism such as AutoEncoder in the feature extraction section.

## Setup AWS Instance
Instance Type : g4dn.xlarge
AMI : Ubuntu Server 20.04 LTS (HVM), SSD Volume Type - ami-0ca5c3bd5a268e7db (64 ビット x86) / ami-0ae8c80279572fa66 (64 ビット Arm)


#### Install nvidia-driver
```
sudo apt update
sudo apt install nvidia-driver-460 -y
sudo reboot
```

#### Install docker (and nvidia-docker2)
cf. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo gpasswd -a $USER docker
sudo reboot
```

#### Build Docker Image and run container
```
docker build -t imbalanced-cifar-10 .
docker run --gpus all -it --name imbalanced-cifar-10-container imbalanced-cifar-10:latest bash
```

## Learn
In the following, the command is described on the assumption that it will be executed in the `source` directory.

#### Learning only a normal classifier for non-imbalanced data
```
./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=5000
```

#### Learning only a normal classifier for imbalance data
```
./train.py --coefficient_of_mse=0 --data_num_of_imbalanced_class=2500
```

#### Learning both classifier and AutoEncoder for imbalance data
```
./train.py --data_num_of_imbalanced_class=2500
```

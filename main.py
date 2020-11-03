import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import tqdm
import time
import argparse

from all_model import get_model

def get_args():
    parser = argparse.ArgumentParser(description="CIFAR10 or CIFAR100 Training")
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning Rate")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--resume", "-r", default=None, help="Resume from pre-train model")
    parser.add_argument("--net", default="vgg16", help="Which model")
    parser.add_argument("--num_classes", default=10, type=int, help="set CIFAR10 or 100")
    parser.add_argument("-c", "--count_parameters", action="store_true",)
    return parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0
start_epoch = 0

# Prepare Date

args = get_args()

transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.num_classes == 10:
    train_set = torchvision.datasets.CIFAR10(
        root="data/", train=True, download=True, transform=transforms_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data/", train=False, download=True, transform=transforms_test
    )
else:
    train_set = torchvision.datasets.CIFAR100(
        root="data/", train=True, download=True, transform=transforms_train
    )
    test_set = torchvision.datasets.CIFAR100(
        root="data/", train=False, download=True, transform=transforms_test
    )

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False, num_workers=0
)

if args.net:
    print(f"Loading Net {args.net} ...")
    net = get_model(args.net, num_classes=args.num_classes)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

classes = train_set.classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def val(epoch, t_epoch):
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    pbar = tqdm.tqdm(test_loader)
    for batch, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs)
        loss = criterion(output, targets)
        val_loss += loss.item()
        _, predict_maxidx = output.max(1)
        correct += predict_maxidx.eq(targets).sum().item()
        total += inputs.size(0)
        pbar.set_description("Epoch %3d/%3d - Val_Loss : %.3f | Val_Acc: %.3f%%" % (epoch, t_epoch, val_loss/(batch+1), 100.*correct/total))


def train(epoch, t_epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm.tqdm(train_loader)
    
    for batch, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_description("Epoch %3d/%3d - Train_Loss : %.3f | Train_Acc: %.3f%%" % (epoch, t_epoch, train_loss/(batch+1), 100.*correct/total))


if __name__ == "__main__":
    if args.count_parameters:
        print('# Parameters:', sum(param.numel() for param in net.parameters()))
    else:
        for epoch in range(args.epochs):
            train(epoch, args.epochs)
            val(epoch, args.epochs)
    # a, b = train_set.__getitem__(1)
    # print(b)
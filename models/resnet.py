import torch
import torch.nn as nn
import torch.nn.functional as F

class  BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel * self.expansion:
            self.shortcut = nn.Sequential(  
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel*self.expansion:
            self.shortcut = nn.Sequential(  
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel*self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
    
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channel, block_num, stride):
        strides = [stride] + [1] * (block_num - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channel, stride=stride))
            self.in_channel = channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

cfg = {
    "18": [2, 2, 2, 2],
    "34": [3, 4, 6, 3],
    '50': [3, 4, 6, 3],
    '101': [3, 4, 23, 3],
    '152': [3, 8, 36, 3]
}

def RESNET(model_type, num_classes):
    if model_type == "18" or model_type == "34":
        block = BasicBlock
    elif model_type == "50" or model_type == "101" or model_type == "152":
        block = Bottleneck

    return ResNet(block, cfg[model_type], num_classes)

if __name__ == "__main__":
    model = RESNET("152", 100)
    img = torch.rand([2, 3, 32, 32])
    print(model(img).shape)

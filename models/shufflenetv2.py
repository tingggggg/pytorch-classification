import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.shuffle = ShuffleBlock()

    def forward(self, x):
        x1, x2 = self.split(x)
        out = F.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.bn3(self.conv3(out)))
        out = torch.cat([x1, out], 1)
        out = self.shuffle(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1_l = nn.Conv2d(in_channels, in_channels, stride=2, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.bn1_l = nn.BatchNorm2d(in_channels)
        self.conv2_l = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2_l = nn.BatchNorm2d(mid_channels)

        self.conv1_r = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1_r = nn.BatchNorm2d(mid_channels)
        self.conv2_r = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn2_r = nn.BatchNorm2d(mid_channels)
        self.conv3_r = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3_r = nn.BatchNorm2d(mid_channels)

        self.shuffle = ShuffleBlock()

    def forward(self, x):
        # left
        out_l = self.bn1_l(self.conv1_l(x))
        out_l = F.relu(self.bn2_l(self.conv2_l(out_l)))
        # right
        out_r = F.relu(self.bn1_r(self.conv1_r(x)))
        out_r = self.bn2_r(self.conv2_r(out_r))
        out_r = F.relu(self.bn3_r(self.conv3_r(out_r)))
        # concat
        out = torch.cat([out_l, out_r], 1)
        out = self.shuffle(out)
        return out
        

class ShuffleNetV2(nn.Module):
    configs = {
        0.5: {
            'out_channels': (48, 96, 192, 1024),
            'num_blocks': (3, 7, 3)
        },

        1: {
            'out_channels': (116, 232, 464, 1024),
            'num_blocks': (3, 7, 3)
        },
        1.5: {
            'out_channels': (176, 352, 704, 1024),
            'num_blocks': (3, 7, 3)
        },
        2: {
            'out_channels': (224, 488, 976, 2048),
            'num_blocks': (3, 7, 3)
        }
    }
    def __init__(self, num_classes=10, net_size=0.5):
        super(ShuffleNetV2, self).__init__()
        out_channels = self.configs[net_size]['out_channels']
        num_blocks = self.configs[net_size]['num_blocks']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv2d(out_channels[2], out_channels[3],
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.linear = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    img = torch.randn([2, 3, 32, 32])
    model = ShuffleNetV2()
    print(model(img).shape)
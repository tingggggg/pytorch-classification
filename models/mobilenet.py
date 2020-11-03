import torch 
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    '''
        Depthwise conv + Pointwise conv
    '''
    def __init__(self, in_channel, out_channel, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)

        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    # (128,2) means conv outchannel=128, conv stride=2
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channel=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_channel):
        layers = []
        for x in self.cfg:
            out_channel = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_channel, out_channel, stride=stride))
            in_channel = out_channel
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    net = MobileNet()
    img = torch.rand([4, 3, 32, 32])
    net(img)



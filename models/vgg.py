import  torch
import torch.nn as nn

cfg = {
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_type, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_type])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3

        for layer_para in cfg:
            if layer_para != "M":
                layers += [
                    nn.Conv2d(in_channels, layer_para, kernel_size=3, padding=1),
                    nn.BatchNorm2d(layer_para),
                    nn.ReLU(inplace=True)
                ]
                in_channels = layer_para
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]        
        return nn.Sequential(*layers)


if __name__ == "__main__":
    img = torch.rand([6, 3, 32, 32])
    print(img.shape)
    model = VGG("VGG11")
    x = model(img)

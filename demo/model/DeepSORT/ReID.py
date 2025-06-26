import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if self.is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

        if self.is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)

        if self.is_downsample:
            x = self.downsample(x)
        
        return F.relu(x.add(y), inplace=True)


def make_layer(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if not i:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out, is_downsample=False), ]
    return nn.Sequential(*blocks)


class ReID(nn.Module):
    def __init__(self, num_classes=751, reid=False):    # 751为Market1501训练集的类别数
        super(ReID, self).__init__()
        self.reid = reid

        # (3, 128, 64) -> (64, 64, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # (64, 64, 32) -> (64, 64, 32)
        self.layer1 = make_layer(64, 64, 2, is_downsample=False)
        # (64, 64, 32) -> (128, 32, 16)
        self.layer2 = make_layer(64, 128, 2, is_downsample=True)
        # (128, 32, 16) -> (256, 16, 8)
        self.layer3 = make_layer(128, 256, 2, is_downsample=True)
        # (256, 16, 8) -> (512, 8, 4)
        self.layer4 = make_layer(256, 512, 2, is_downsample=True)
        # (512, 8, 4) -> (512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # 提取特征
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        
        # 进行分类任务
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    net = ReID()
    print(net)
    x = torch.randn(32, 3, 128, 64)
    y = net(x)
    print(y.shape)
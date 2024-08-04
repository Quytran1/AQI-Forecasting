import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile


class Resnet18_3d(nn.Module):
    def __init__(self):
        super(Resnet18_3d, self).__init__()

        # lớp đầu tiên
        self.conv1 = nn.Conv3d(
            in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2
        )
        self.bn1 = nn.BatchNorm3d(num_features=64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Các lớp tiếp theo của ResNet được giữ nguyên nhưng với Conv3d
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Lớp Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # Lớp Fully Connected
        self.fc = nn.Linear(512, 1)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.maxpool(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return out


if __name__ == "__main__":
    fake_data = torch.randn(
        10, 3, 3, 224, 224
    )  # (batch_size, times, channels,  height, width)

    model = Resnet18_3d()

    output = model(fake_data)
    print(output.shape)
    # summary(model, fake_data.shape)
    # flops, params = profile(model, inputs=(fake_data,))

    # In FLOPs và số lượng tham số
    # print(f"FLOPs: {flops}, Params: {params}")

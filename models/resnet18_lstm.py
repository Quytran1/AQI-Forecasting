import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile

# class Resnet18lstm(nn.Module):
#     def __init__(self):
#         super(Resnet18lstm, self).__init__()

#         # lớp đầu tiên
#         self.conv1 = nn.Conv1d(
#             in_channels=8, out_channels=64, kernel_size=7, padding=3, stride=2
#         )
#         self.bn1 = nn.BatchNorm1d(num_features=64)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#         # Lớp thứ 2 gồm 2 block mỗi block có 2 layer conv1d k=3 đầu vào và đầu ra là 64
#         self.conv2_1 = nn.Conv1d(
#             in_channels=64, out_channels=64, kernel_size=3, padding=1
#         )
#         self.bn2_1 = nn.BatchNorm1d(num_features=64)
#         self.conv2_2 = nn.Conv1d(
#             in_channels=64, out_channels=64, kernel_size=3, padding=1
#         )
#         self.bn2_2 = nn.BatchNorm1d(num_features=64)

#         # Lớp thứ 3 gồm 2 block mỗi block có 2 layer conv1d k=3 đầu vào và đầu ra là 64
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.conv3_1 = nn.Conv1d(
#             in_channels=128, out_channels=128, kernel_size=3, padding=1
#         )
#         self.bn3_1 = nn.BatchNorm1d(num_features=128)
#         self.conv3_shortcut = nn.Conv1d(64, 128, kernel_size=1, stride=2, padding=0)
#         self.bn3_shortcut = nn.BatchNorm1d(128)

#         # Lớp thứ 4 gồm 2 block mỗi block có 2 layer conv1d k=3 đầu vào và đầu ra là 64
#         self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.conv4_1 = nn.Conv1d(
#             in_channels=256, out_channels=256, kernel_size=3, padding=1
#         )
#         self.bn4_1 = nn.BatchNorm1d(num_features=256)
#         self.conv4_shortcut = nn.Conv1d(128, 256, kernel_size=1, stride=2, padding=0)
#         self.bn4_shortcut = nn.BatchNorm1d(256)

#         # Lớp thứ 5 gồm 2 block mỗi block có 2 layer conv1d k=3 đầu vào và đầu ra là 64
#         self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
#         self.bn5 = nn.BatchNorm1d(512)
#         self.conv5_1 = nn.Conv1d(
#             in_channels=512, out_channels=512, kernel_size=3, padding=1
#         )
#         self.bn5_1 = nn.BatchNorm1d(num_features=512)
#         self.conv5_shortcut = nn.Conv1d(256, 512, kernel_size=1, stride=2, padding=0)
#         self.bn5_shortcut = nn.BatchNorm1d(512)

#         # Lớp LSTM
#         self.lstm = nn.LSTM(512, 512, batch_first=True)
#         # Lớp Global Average Pooling
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         # Lớp Fully Connected
#         self.fc = nn.Linear(512, 1)

#     def forward(self, x):

#         # Dữ liệu đi qua lớp đầu tiên
#         x = F.relu(self.maxpool(self.bn1(self.conv1(x))))
#         # print("layer1: ", x.shape)

#         # Dữ liệu đi qua lớp thứ 2
#         # Block 1
#         shortcut = x
#         print(shortcut.shape)
#         x = F.relu(self.bn2_1(self.conv2_1(x)))
#         x = F.relu(self.bn2_2(self.conv2_2(x)))
#         x = x + shortcut
#         print("layer1 block1: ", x.shape)
#         # Block 2
#         shortcut = x
#         x = F.relu(self.bn2_1(self.conv2_1(x)))
#         x = F.relu(self.bn2_2(self.conv2_2(x)))
#         x = x + shortcut
#         print("layer2 block2: ", x.shape)

#         # Dữ liệu đi qua lớp thứ 3
#         shortcut = x
#         x = F.relu(self.bn3(self.conv3(x)))
#         print(x.shape)
#         x = F.relu(self.bn3_1(self.conv3_1(x)))
#         shortcut = F.relu(self.bn3_shortcut(self.conv3_shortcut(shortcut)))
#         x = x + shortcut
#         shortcut = x
#         x = F.relu(self.bn3_1(self.conv3_1(x)))
#         x = F.relu(self.bn3_1(self.conv3_1(x)))
#         x = x + shortcut
#         print(x.shape)

#         # Dữ liệu đi qua lớp thứ 4
#         shortcut = x
#         x = F.relu(self.bn4(self.conv4(x)))
#         print(x.shape)
#         x = F.relu(self.bn4_1(self.conv4_1(x)))
#         shortcut = F.relu(self.bn4_shortcut(self.conv4_shortcut(shortcut)))
#         x = x + shortcut
#         shortcut = x
#         x = F.relu(self.bn4_1(self.conv4_1(x)))
#         x = F.relu(self.bn4_1(self.conv4_1(x)))
#         x = x + shortcut
#         print(x.shape)

#         # Dữ liệu đi qua lớp thứ 5
#         shortcut = x
#         x = F.relu(self.bn5(self.conv5(x)))
#         print(x.shape)
#         x = F.relu(self.bn5_1(self.conv5_1(x)))
#         shortcut = F.relu(self.bn5_shortcut(self.conv5_shortcut(shortcut)))
#         x = x + shortcut
#         shortcut = x
#         x = F.relu(self.bn5_1(self.conv5_1(x)))
#         x = F.relu(self.bn5_1(self.conv5_1(x)))
#         x = x + shortcut
#         print(x.shape)

#         # Dữ liệu đi qua lớp lstm
#         x = x.permute(0, 2, 1)

#         x, (hn, cn) = self.lstm(x)
#         x = self.avgpool(x.transpose(1, 2))
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x


class Resnet18lstm(nn.Module):
    def __init__(self):
        super(Resnet18lstm, self).__init__()

        # lớp đầu tiên
        self.conv1 = nn.Conv1d(
            in_channels=8, out_channels=64, kernel_size=7, padding=3, stride=2
        )
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Các lớp tiếp theo của ResNet được giữ nguyên nhưng với Conv1d
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Lớp LSTM
        self.lstm = nn.LSTM(512, 512, batch_first=True)
        # Lớp Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
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

        x = x.permute(0, 2, 1)

        x, (hn, cn) = self.lstm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        # out = F.relu(out)
        return out


if __name__ == "__main__":
    fake_data = torch.randn(10, 24, 8)
    x = fake_data.permute(0, 2, 1)

    model = Resnet18lstm()

    output = model(x)
    summary(model, x.shape)
    flops, params = profile(model, inputs=(x,))

    # In FLOPs và số lượng tham số
    print(f"FLOPs: {flops}, Params: {params}")


# độ phức tạp của mô hình(flops"chi phí tính toán")
# số lượng tham số của resnet, inceptionet, bilstm
# inception netv3, v4
# googlenet
# mobinet, shuffenet, ..

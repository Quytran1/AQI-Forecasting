import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionA, self).__init__()
        self.conv1x1 = BasicConv1d(in_channels, 96, kernel_size=1)

        self.conv3x1_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.conv3x1_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)

        self.conv_pool = BasicConv1d(in_channels, out_channels, kernel_size=1)

        self.conv3x1dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.conv3x1dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.conv3x1dbl_3 = BasicConv1d(96, 96, kernel_size=3, padding=1)

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        print(conv1x1.shape)
        conv5x5 = self.conv3x1_1(x)
        conv5x5 = self.conv3x1_2(conv5x5)
        print(conv5x5.shape)
        conv3x1dbl = self.conv3x1dbl_1(x)
        conv3x1dbl = self.conv3x1dbl_2(conv3x1dbl)
        conv3x1dbl = self.conv3x1dbl_3(conv3x1dbl)
        print(conv3x1dbl.shape)
        conv_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        conv_pool = self.conv_pool(conv_pool)
        print(conv_pool.shape)
        outputs = [conv1x1, conv5x5, conv3x1dbl, conv_pool]
        return torch.cat(outputs, 1)


class GridReductionA(nn.Module):
    def __init__(self, in_channels):
        super(GridReductionA, self).__init__()
        self.conv3x1 = BasicConv1d(384, 384, kernel_size=3, stride=2)

        self.conv3x1dbl_1 = BasicConv1d(384, 64, kernel_size=1)
        self.conv3x1dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.conv3x1dbl_3 = BasicConv1d(96, 256, kernel_size=3, stride=2)

        self.conv_pool = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        conv3x1 = self.conv3x1(x)

        conv3x1dbl = self.conv3x1dbl_1(x)
        conv3x1dbl = self.conv3x1dbl_2(conv3x1dbl)
        conv3x1dbl = self.conv3x1dbl_3(conv3x1dbl)

        conv_pool = self.conv_pool(x)

        outputs = [conv3x1, conv3x1dbl, conv_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        self.branch1x1 = BasicConv1d(1024, 384, kernel_size=1)

        self.branch7x1_1 = BasicConv1d(1024, 192, kernel_size=1)
        self.branch7x1_2 = BasicConv1d(192, 256, kernel_size=7, padding=3)

        self.branch7x1dbl_1 = BasicConv1d(1024, 192, kernel_size=1)
        self.branch7x1dbl_2 = BasicConv1d(192, 224, kernel_size=7, padding=3)
        self.branch7x1dbl_3 = BasicConv1d(224, 256, kernel_size=7, padding=3)

        self.branch_pool = nn.Conv1d(1024, 128, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x1 = self.branch7x1_1(x)
        branch7x1 = self.branch7x1_2(branch7x1)

        branch7x1dbl = self.branch7x1dbl_1(x)
        branch7x1dbl = self.branch7x1dbl_2(branch7x1dbl)
        branch7x1dbl = self.branch7x1dbl_3(branch7x1dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x1, branch7x1dbl, branch_pool]
        return torch.cat(outputs, 1)


class GridReductionB(nn.Module):
    def __init__(self, in_channels):
        super(GridReductionB, self).__init__()
        self.branch7x1x3_1 = BasicConv1d(1024, 256, kernel_size=1)
        self.branch7x1x3_2 = BasicConv1d(256, 256, kernel_size=7, padding=3)
        self.branch7x1x3_3 = BasicConv1d(256, 320, kernel_size=3, stride=2)

        self.branch3x1_1 = BasicConv1d(in_channels, 192, kernel_size=1)
        self.branch3x1_2 = BasicConv1d(192, 192, kernel_size=3, stride=2)

        self.branch_pool = nn.MaxPool1d(kernel_size=3, stride=2)

    def forward(self, x):
        branch7x1x3 = self.branch7x1x3_1(x)
        branch7x1x3 = self.branch7x1x3_2(branch7x1x3)
        branch7x1x3 = self.branch7x1x3_3(branch7x1x3)

        branch3x1 = self.branch3x1_1(x)
        branch3x1 = self.branch3x1_2(branch3x1)

        branch_pool = self.branch_pool(x)

        outputs = [branch7x1x3, branch3x1, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv1d(1536, 256, kernel_size=1)

        self.branch3x1_1 = BasicConv1d(in_channels, 384, kernel_size=1)
        self.branch3x1_2a = BasicConv1d(384, 256, kernel_size=3, padding=1)
        self.branch3x1_2b = BasicConv1d(384, 256, kernel_size=3, padding=1)

        self.branch3x1dbl_1 = BasicConv1d(in_channels, 448, kernel_size=1)
        self.branch3x1dbl_2 = BasicConv1d(448, 512, kernel_size=3, padding=1)
        self.branch3x1dbl_3a = BasicConv1d(512, 256, kernel_size=3, padding=1)
        self.branch3x1dbl_3b = BasicConv1d(512, 256, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv1d(1536, 256, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x1 = self.branch3x1_1(x)
        branch3x1 = torch.cat(
            (self.branch3x1_2a(branch3x1), self.branch3x1_2b(branch3x1)), 1
        )

        branch3x1dbl = self.branch3x1dbl_1(x)
        branch3x1dbl = self.branch3x1dbl_2(branch3x1dbl)
        branch3x1dbl = torch.cat(
            (self.branch3x1dbl_3a(branch3x1dbl), self.branch3x1dbl_3b(branch3x1dbl)), 1
        )

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x1, branch3x1dbl, branch_pool]
        return torch.cat(outputs, 1)


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()

        self.conv1 = BasicConv1d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv1d(32, 64, kernel_size=3, padding=1)

        self.branch1_maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.branch2 = BasicConv1d(64, 96, kernel_size=3, stride=2, padding=1)

        self.branch3_1 = BasicConv1d(160, 64, kernel_size=1)
        self.branch3_2 = BasicConv1d(64, 64, kernel_size=7, padding=3)
        self.branch3_3 = BasicConv1d(64, 96, kernel_size=3, padding=1)

        self.branch4_1 = BasicConv1d(160, 64, kernel_size=1)
        self.branch4_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)

        self.branch5_maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.branch6 = BasicConv1d(192, 192, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([self.branch1_maxpool(x), self.branch2(x)], 1)
        x1 = self.branch3_1(x)
        x1 = self.branch3_2(x1)
        x1 = self.branch3_3(x1)
        x2 = self.branch4_1(x)
        x2 = self.branch4_2(x2)
        x = torch.cat([x1, x2], 1)
        x = torch.cat([self.branch5_maxpool(x), self.branch6(x)], 1)
        return x


class InceptionNetV4(nn.Module):
    def __init__(self):
        super(InceptionNetV4, self).__init__()
        self.stem = Stem(8)
        self.inception_a = self._make_inception_a(4)
        self.reduction_a = GridReductionA(384)
        self.inception_b = self._make_inception_b(7)
        self.reduction_b = GridReductionB(1024)
        self.inception_c = self._make_inception_c(3)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(1536, 1)

    def _make_inception_a(self, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(InceptionA(384, 96))
        return nn.Sequential(*layers)

    def _make_inception_b(self, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(InceptionB(1024, 1024))
        return nn.Sequential(*layers)

    def _make_inception_c(self, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(InceptionC(1536, 1536))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = InceptionNetV4()
    input_data = torch.randn(10, 8, 112)
    # x = input_data.permute(0, 2, 1)
    output = model(input_data)
    print(output.shape)
    summary(model, input_data.shape)
    flops, params = profile(model, inputs=(input_data,))

    # In FLOPs và số lượng tham số
    print(f"FLOPs: {flops}, Params: {params}")

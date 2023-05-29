import torch
from torch import nn
import torch.nn.functional as F


class AddLaplacian(nn.Module):
    def __init__(self):
        super(AddLaplacian, self).__init__()
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float).view(1, 1, 3, 3)

        self._kernel = torch.nn.Parameter(kernel.repeat(1, 3, 1, 1),
                                          requires_grad=False)

    def forward(self, x):
        y = F.conv2d(x, self._kernel, stride=1, padding=1, )
        y = torch.cat([x, y], dim=1)
        return y


class LaplacianLayer(nn.Module):
    def __init__(self):
        super(LaplacianLayer, self).__init__()
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float).view(1, 1, 3, 3)

        self._kernel = torch.nn.Parameter(kernel.repeat(1, 3, 1, 1),
                                          requires_grad=False)

    def forward(self, x):
        y = F.conv2d(x, self._kernel, stride=1, padding=1, )
        return y


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels, strides=1, use_1x1conv=False):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResidualBlocks(nn.Module):
    def __init__(self, num_blocks, num_channels):
        super(ResidualBlocks, self).__init__()
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module("ResidualBlock%d" % (i+1), ResidualBlock(num_channels, num_channels))

    def forward(self, x):
        return self.blocks(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rgb_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.laplacian_block = nn.Sequential(
            LaplacianLayer(),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # self.mix_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64), nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(128), nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        # self.output_layer = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(128, 56),
        #     nn.BatchNorm1d(56),
        #     nn.ReLU(),
        #     nn.Linear(56, 2)
        # )
        self.mix_layer = ResidualBlocks(num_blocks=3, num_channels=32)
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        y_laplacian = self.laplacian_block(x)
        y_rgb = self.rgb_block(x)
        #
        x_mix = torch.cat([y_laplacian, y_rgb], dim=1)
        y_mix = self.mix_layer(x_mix)
        output = self.output_layer(y_mix)

        return output


if __name__ == '__main__':
    from torchsummary import summary
    from torchstat import stat
    # model = ModelText()
    #
    # model = model.to(torch.device('cuda'))
    # summary(model, input_size=(3, 448, 448))
    #
    # model.cpu()
    # print("")
    # stat(model, (3, 448, 448))

    # params = list(model.parameters())
    # print("")
    # print(params[0])
    # p = list(model.model[0].named_modules())
    # print(p[0])
    # add = AddLaplacian()
    # x = torch.rand((1, 3, 448, 448))
    # print(add(x).shape)

    # model = ModelText2()
    #
    # model = model.to(torch.device('cuda'))
    # summary(model, input_size=(3, 448, 448))

    # ModelText3
    model = Model()
    # stat(model, input_size=(3, 448, 448))
    model = model.to(torch.device('cuda'))
    summary(model, input_size=(3, 448, 448))
    # block = ResidualBlocks(num_blocks=6, num_channels=60)
    # x = torch.randn((1, 60, 66, 66))
    # y = block(x)
    # print(y.shape)
    # print(block)

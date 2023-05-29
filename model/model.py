import torch
from torch import nn
from torchsummary import summary
from torchstat import stat
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


# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = nn.Sequential(
#             AddLaplacian(),
#
#             nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(16), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(512), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(512, 2)
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class ModelText(nn.Module):
#     def __init__(self):
#         super(ModelText, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(16), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(32), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(128, 2)
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class ModelText2(nn.Module):
#     def __init__(self):
#         super(ModelText2, self).__init__()
#         self.model = nn.Sequential(
#             LaplacianLayer(),
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(16), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(32), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(128, 2)
#         )
#
#     def forward(self, x):
#         return self.model(x)


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
        self.mix_layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 56),
            nn.BatchNorm1d(56),
            nn.ReLU(),
            nn.Linear(56, 2)
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


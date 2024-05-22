"""
The LeNet model used in the Zhu's implementation.

Reference:
Zhu et al., "Deep Leakage from Gradients," in the Proceedings of NeurIPS 2019.
https://github.com/mit-han-lab/dlg
"""
import torch.nn as nn
from plato.config import Config

class Model(nn.Module):
    def __init__(self, num_classes=Config().parameters.model.num_classes):
        super(Model, self).__init__()
        act = nn.Sigmoid
        # if Config().data.datasource == "EMNIST":
        if Config().data.datasource == "FashionMNIST":
            in_channel = 1
            in_size = 588
        if Config().data.datasource.startswith("CIFAR"):
            in_channel = 3
            in_size = 768
        self.conv1 = nn.Conv2d(in_channel, 12, kernel_size=5, padding=5 // 2, stride=2)
        self.act1 = act()
        self.conv2 = nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2)
        self.act2 = act()
        self.conv3 = nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1)
        self.act3 = act()
        self.fc = nn.Linear(in_size, num_classes)

    def forward(self, x):
        fea_list = []
        out = self.conv1(x)
        out = self.act1(out)
        fea_list.append(out)  # 第一个中间层的输出
        out = self.conv2(out)
        out = self.act2(out)
        fea_list.append(out)  # 第二个中间层的输出
        out = self.conv3(out)
        out = self.act3(out)
        fea_list.append(out)  # 第三个中间层的输出

        feature = out.view(out.size(0), -1)
        out = self.fc(feature)
        return out, feature, fea_list

# class Model(nn.Module):
#     def __init__(self, num_classes=Config().parameters.model.num_classes):
#         super().__init__()
#         act = nn.Sigmoid
#         if Config().data.datasource == "EMNIST":
#             in_channel = 1
#             in_size = 588
#         if Config().data.datasource.startswith("CIFAR"):
#             in_channel = 3
#             in_size = 768
#
#         self.body = nn.Sequential(
#             nn.Conv2d(in_channel, 12, kernel_size=5, padding=5 // 2, stride=2),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
#             act(),
#         )
#         self.fc = nn.Sequential(nn.Linear(in_size, num_classes))
#
#     def forward(self, x):
#         out = self.body(x)
#         feature = out.view(out.size(0), -1)
#         out = self.fc(feature)
#
#         return out, feature

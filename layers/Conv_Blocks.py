import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        # 并行self.num_kernels个卷积
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print("#################################### Inception-1 #####################################")
        # print(self.in_channels)                                 # 16
        # print(self.out_channels)                                # 32
        # print(self.num_kernels)                                 # 6
        res = None
        for i in range(self.num_kernels):
            # print(i)
            if res == None:
                res = self.kernels[i](x)/self.num_kernels
            else:
                res = res + (self.kernels[i](x)/self.num_kernels)
        # print(res.shape)                                        # torch.Size([32, 32, 48, 4])
        # print("#################################### Inception-2 #####################################")
        return res
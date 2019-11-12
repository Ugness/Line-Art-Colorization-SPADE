import torch.nn as nn


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode='zeros')
        self.pad = nn.ReflectionPad2d(padding=padding)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d_forward(x, self.weight)
        return x


if __name__ == "__main__":
    import torch

    a = Conv2d(1, 3, 3, padding=1)
    b = a(torch.zeros([1, 1, 12, 12]))
    print(b.shape)

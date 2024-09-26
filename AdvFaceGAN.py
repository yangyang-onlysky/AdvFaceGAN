import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False,
            padding_mode='reflect'
        )
        self.instanceNorm2d = nn.InstanceNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.instanceNorm2d(x)
        if self.activation:
            x = self.activation(x)
        return x


class BasicDeConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, activation=None):
        super().__init__()
        self.deConv = nn.ConvTranspose2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False,
        )
        self.instanceNorm2d = nn.InstanceNorm2d(out_planes)
        self.activation = activation

    def forward(self, x):
        x = self.deConv(x)
        x = self.instanceNorm2d(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       nn.BatchNorm2d(dim, eps=0.001, momentum=0.1, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                       nn.BatchNorm2d(dim, eps=0.001, momentum=0.1, affine=True),]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        out = F.relu(out, inplace=True)
        return out


# Generator Network
class Generator(nn.Module):
    def __init__(self, is_target=False):
        super(Generator, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        k = 64
        self.conv1 = BasicConv2d(6 if is_target else 3, k, 7, stride=1, activation=self.lrelu)
        self.conv2 = BasicConv2d(k, 2 * k, 4, stride=2, activation=self.lrelu)
        self.conv3 = BasicConv2d(2 * k, 4 * k, 4, stride=2, activation=self.lrelu)
        self.res_blocks = nn.Sequential(
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
            ResnetBlock(4 * k),
        )
        self.deconv1 = BasicDeConv2d(4 * k, 2 * k, 4, stride=2, activation=self.lrelu)
        self.deconv2 = BasicDeConv2d(2 * k, k, 4, stride=2, activation=self.lrelu)
        self.conv_img = nn.ConvTranspose2d(k, 3, 7, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, sources, targets=None):
        if targets is not None:
            net = torch.cat((sources, targets), dim=1)
        else:
            net = sources
        # 编码
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.conv3(net)
        # 残差连接
        encoded = self.res_blocks(net)
        # 解码
        net = self.deconv1(encoded)
        net = self.deconv2(net)
        net = self.conv_img(net)
        perturb = self.tanh(net)
        # 加性扰动
        output = 2 * torch.clip(perturb + (sources + 1.0) / 2.0, 0, 1) - 1
        return perturb, output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Discriminator Network
class ResNetDiscriminator(nn.Module):
    def __init__(self):
        super(ResNetDiscriminator, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.fc = nn.Linear(512 * 14 * 14, 1)  # 输出一个值，用于表示真实性

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.in1(self.layer1(x)))
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, device, init_type='normal', init_gain=0.02):
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net

# 单元测试
import unittest


class TestModel(unittest.TestCase):
    def test_generator_forward(self):
        source_tensor = torch.randn(10, 3, 112, 112)
        target_tensor = torch.randn(10, 3, 112, 112)
        model = Generator(is_target=True)
        pertub, output = model(source_tensor, target_tensor)
        self.assertEqual((10, 3, 112, 112), output.shape)

    def test_generator_backward(self):
        source_tensor = torch.randn(10, 3, 112, 112).requires_grad_()
        target_tensor = torch.randn(10, 3, 112, 112)
        model = Generator(is_target=False)
        pertub, output = model(sources=source_tensor)
        output.sum().backward()
        self.assertIsNotNone(source_tensor.grad)

    def test_discriminator_forward(self):
        model = ResNetDiscriminator()
        input_tensor = torch.randn(10, 3, 112, 112)
        output = model(input_tensor)
        self.assertEqual((10, 1), output.shape)

    def test_backward(self):
        model = ResNetDiscriminator()
        input_tensor = torch.randn(10, 3, 112, 112, requires_grad=True)
        output = model(input_tensor)
        output.sum().backward()
        self.assertIsNotNone(input_tensor.grad)


if __name__ == '__main__':
    unittest.main()

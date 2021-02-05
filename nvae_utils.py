import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SwishFunction(torch.autograd.Function):
    # class of swish activation function
    @staticmethod
    def forward(ctx, x_in):
        ctx.save_for_backward(x_in)
        out = x_in * torch.sigmoid(x_in)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_in, = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x_in)
        grad_input = grad_output * (sigmoid_x * (1 + x_in - x_in * sigmoid_x))
        return grad_input


class Swish(nn.Module):
    # call of swish activation function (as a layer)
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishFunction.apply(x)


class SELayer(nn.Module):
    # class for Squeeze and Excitation Layer
    def __init__(self, num_channels, reduction_ratio=16):
        super(SELayer, self).__init__()
        num_hidden = max(num_channels // reduction_ratio, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(  # TODO bias = True or False
            nn.Linear(num_channels, num_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(num_hidden, num_channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # input size (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        se = self.squeeze(x)  # size (batch_size, channels, 1, 1)
        se = se.view(batch_size, channels)  # size (batch_size, channels)
        se = self.excitation(se)
        se = se.view(batch_size, channels, 1, 1)  # size (batch_size, channels, 1, 1)
        return x * se


class Encoder_Residual_Cell(nn.Module):  # TODO seq_weight, momentum definition, conv size
    def __init__(self, num_channels, seq_weight=0.1):
        super(Encoder_Residual_Cell, self).__init__()
        self.seq_weight = seq_weight
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_channels, momentum=0.05),
            Swish(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels, momentum=0.05),
            Swish(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            SELayer(num_channels),
        )

    def forward(self, x):
        return x + self.seq_weight * self.seq(x)


class Decoder_Residual_Cell(nn.Module):  # TODO seq_weight, momentum definition, conv size
    def __init__(self, num_channels, up_times=2, seq_weight=0.1):
        super(Decoder_Residual_Cell, self).__init__()
        num_up_channels = num_channels * up_times
        self.seq_weight = seq_weight
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_channels, momentum=0.05),
            nn.Conv2d(num_channels, num_up_channels, kernel_size=1),  # up sample
            nn.BatchNorm2d(num_up_channels, momentum=0.05),
            Swish(),

            # depthwise convolution
            nn.Conv2d(num_up_channels, num_up_channels, kernel_size=5, padding=2, groups=num_up_channels),

            nn.BatchNorm2d(num_up_channels, momentum=0.05),
            Swish(),
            nn.Conv2d(num_up_channels, num_channels, kernel_size=1),  # down sample
            nn.BatchNorm2d(num_channels, momentum=0.05),
            SELayer(num_channels),
        )

    def forward(self, x):
        return x + self.seq_weight * self.seq(x)


class ConvBlock(nn.Module):
    def __init__(self, num_in_channel, num_out_channel):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_in_channel, num_out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_out_channel, momentum=0.05),
            Swish(),
        )

    def forward(self, x):
        return self.conv(x)


class ConvTranBlock(nn.Module):
    def __init__(self, num_in_channel, num_out_channel):
        super(ConvTranBlock, self).__init__()
        self.convtran = nn.Sequential(
            nn.ConvTranspose2d(num_in_channel, num_out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_out_channel, momentum=0.05),
            Swish(),
        )

    def forward(self, x):
        return self.convtran(x)


class Combiner(nn.Module):
    def __init__(self, num_channel, combiner_type):
        super(Combiner, self).__init__()
        if combiner_type == 'enc':
            self.conv = nn.Conv2d(2*num_channel, 2*num_channel, kernel_size=1, stride=1)
        else:
            self.conv = nn.Conv2d(2*num_channel, num_channel, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        out = torch.cat((x1, x2), dim=1)
        out = self.conv(out)
        return out


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    z = mu + std * torch.randn_like(mu, device=DEVICE)
    return z


def kl(mu, logvar):
    kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return kld


def kl_rela(logvar, delta_mu, delta_logvar):
    kld = 0.5 * torch.sum(delta_mu.pow(2) / logvar.exp() + delta_logvar.exp() - delta_logvar - 1)
    return kld

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SwishFunction(torch.autograd.Function):
    """
    Class of swish activation function.
    """
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
    """
    Call of swish activation function (as a layer).
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishFunction.apply(x)


class SELayer(nn.Module):
    """
    Class for Squeeze and Excitation Layer.
    """
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


class EncoderResidualCell(nn.Module):  # TODO seq_weight, momentum definition, conv size
    """
    Class for residual cell in encoder.\n
    Structure: BN - Swish - Conv - BN - Swish - Conv - SE
    """
    def __init__(self, num_channels, seq_weight):
        super(EncoderResidualCell, self).__init__()
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


class DecoderResidualCell(nn.Module):  # TODO seq_weight, momentum definition, conv size
    """
    Class for residual cell in encoder.\n
    Structure: BN - Conv - BN - Swish - dep.Conv - BN - Swish - BN - SE
    """
    def __init__(self, num_channels, up_times, seq_weight):
        super(DecoderResidualCell, self).__init__()
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
    """
    Class for convolutional block.\n
    Structure: conv - BN - Swish
    """
    def __init__(self, num_in_channel, num_out_channel):
        super(ConvBlock, self).__init__()
        if num_in_channel == num_out_channel:
            self.conv = nn.Sequential(
                nn.Conv2d(num_in_channel, num_out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_out_channel, momentum=0.05),
                Swish(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(num_in_channel, num_out_channel, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(num_out_channel, momentum=0.05),
                Swish(),
            )

    def forward(self, x):
        return self.conv(x)


class ConvTranBlock(nn.Module):
    """
    Class for transposed convolutional block.\n
    Structure: ConvTranspose - BN - Swish
    """
    def __init__(self, num_in_channel, num_out_channel):
        super(ConvTranBlock, self).__init__()
        if num_in_channel == num_out_channel:
            self.convtran = nn.Sequential(
                nn.ConvTranspose2d(num_in_channel, num_out_channel, kernel_size=3, stride=1, padding=1, output_padding=0),
                nn.BatchNorm2d(num_out_channel, momentum=0.05),
                Swish(),
            )
        else:
            self.convtran = nn.Sequential(
                nn.ConvTranspose2d(num_in_channel, num_out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(num_out_channel, momentum=0.05),
                Swish(),
            )

    def forward(self, x):
        return self.convtran(x)


class EncoderBlock(nn.Module):
    """
    Class for encoder block.\n
    Structure: Input - ConvBlock - EncoderResidual - ConvBlock - EncoderResidual - ...
    """
    def __init__(self, num_in_channel, num_out_channel, num_residual_cell, seq_weight=0.1):
        super(EncoderBlock, self).__init__()
        self.module_list = [ConvBlock(num_in_channel, num_out_channel)]
        for i in range(num_residual_cell):
            self.module_list.append(EncoderResidualCell(num_out_channel, seq_weight))
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class DecoderBlock(nn.Module):
    """
    Class for decoder block.\n
    Structure: Input - ConvTranBlock - DecoderResidual - ConvTranBlock - DecoderResidual - ...
    """
    def __init__(self, num_in_channel, num_out_channel, num_residual_cell, residual_up_times=2, seq_weight=0.1):
        super(DecoderBlock, self).__init__()
        self.module_list = [ConvTranBlock(num_in_channel, num_out_channel)]
        for i in range(num_residual_cell):
            self.module_list.append(DecoderResidualCell(num_out_channel, residual_up_times, seq_weight))
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class Combiner(nn.Module):
    """
    Class for combiners.\n
    Type:\n
    'enc': Channel_in = Channel_out\n
    'dec': Channel_in = Channel_out * 2
    """
    def __init__(self, num_channel, combiner_type):
        super(Combiner, self).__init__()
        if combiner_type == 'enc':
            self.seq = nn.Sequential(
                nn.Conv2d(2 * num_channel, 2 * num_channel, kernel_size=1, stride=1),
                nn.Tanh(),
            )
        else:
            self.seq = nn.Conv2d(2*num_channel, num_channel, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        out = torch.cat((x1, x2), dim=1)
        out = self.seq(out)
        return out


def reparametrize(mu, logvar):
    """
    Sample from a normal distribution.\n
    :param mu: mean of normal distribution
    :param logvar: log variance of normal distribution
    :return: sampled value from normal distribution
    """
    std = logvar.mul(0.5).exp_()
    z = mu + std * torch.randn_like(mu, device=DEVICE)
    return z


def kl(mu, logvar):
    """
    KL-divergence of a distribution and a normal distribution.\n
    :param mu: mean of a distribution
    :param logvar: log variance of a distribution
    :return: KL-divergence
    """
    kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return kld


def kl_rela(logvar, delta_mu, delta_logvar):
    """
    Relative KL-divergence of two distributions base on relative value of mean and log variance.\n
    :param logvar: log variance of a distribution
    :param delta_mu: relative value of mean
    :param delta_logvar: relative value of log variance
    :return: relative KL-divergence
    """
    kld = 0.5 * torch.sum(delta_mu.pow(2) / (logvar.exp() + 1e-7) + delta_logvar.exp() - delta_logvar - 1)
    return kld


def channels_cal(num_in_channel, num_in_size, latent_z_size, double_group):
    """
    Calculates needed groups and channels for layers.\n
    :param num_in_channel: The number of channels of input images.
    :param num_in_size: Size of input images.
    :param latent_z_size: Target size of latent_z.
    :param double_group: Option for doubling groups of latent variables.
    :return:
        num_group: The number of groups of latent variables.
        num_channels_list: List for the numbers of channels in different layers.
    """
    num_max_channel = latent_z_size[0]
    num_min_size = latent_z_size[1:]
    if num_max_channel / 4 < num_in_size[0] / num_min_size[0] or num_max_channel / 4 < num_in_size[1] / num_min_size[1]:
        raise Exception('Expect: latent_z_size[0]/4 >= num_in_size[i]/latent_z_size[i]')

    num_channels_list = []
    num_group = 0
    num_single_group = 0
    num_current_channels = num_max_channel
    while num_min_size[0] * 2 ** num_single_group < num_in_size[0] and num_min_size[1] * 2 ** num_single_group < num_in_size[1]:
        num_channels_list.append(num_current_channels)
        num_single_group += 1
        num_group += 1
        if double_group:
            num_channels_list.append(num_current_channels)
            num_group += 1
        num_current_channels //= 2
    num_channels_list.append(num_in_channel)
    num_channels_list.reverse()
    return num_group, num_channels_list

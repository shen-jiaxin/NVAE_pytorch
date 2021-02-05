import torch
import torch.nn as nn

from nvae_utils import Swish, SELayer, Encoder_Residual_Cell, Decoder_Residual_Cell, \
    ConvBlock, ConvTranBlock, Combiner, reparametrize, kl, kl_rela

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NvaeModel(nn.Module):
    def __init__(self, num_in_channel, num_in_size, batch_size):
        super(Nvae, self).__init__()
        self.conv1 = ConvBlock(num_in_channel, 16)
        self.enc_res1 = Encoder_Residual_Cell(16)
        self.conv2 = ConvBlock(16, 32)
        self.enc_res2 = Encoder_Residual_Cell(32)
        self.conv3 = ConvBlock(32, 64)
        self.enc_res3 = Encoder_Residual_Cell(64)

        self.h = nn.Parameter(torch.randn((batch_size, 64, num_in_size//8, num_in_size//8)).cuda())
        self.upconv1 = ConvTranBlock(64, 32)
        self.dec_res1 = Decoder_Residual_Cell(32)
        self.upconv2 = ConvTranBlock(32, 16)
        self.dec_res2 = Decoder_Residual_Cell(16)
        self.upconv3 = ConvTranBlock(16, num_in_channel)
        self.dec_res3 = Decoder_Residual_Cell(num_in_channel)

        self.enc_combiner1 = Combiner(64, 'enc')
        self.enc_combiner2 = Combiner(32, 'enc')
        self.enc_combiner3 = Combiner(16, 'enc')
        self.dec_combiner1 = Combiner(64, 'dec')
        self.dec_combiner2 = Combiner(32, 'dec')
        self.dec_combiner3 = Combiner(16, 'dec')

    def forward(self, x):
        enc_z3 = self.enc_res1(self.conv1(x))
        enc_z2 = self.enc_res2(self.conv2(enc_z3))
        enc_z1 = self.enc_res3(self.conv3(enc_z2))

        mu1, logvar1 = self.enc_combiner1(enc_z1, enc_z1).chunk(2, dim=1)
        sample1 = reparametrize(mu1, logvar1)
        dec_combined1 = self.dec_combiner1(sample1, self.h)
        dec_z2 = self.dec_res1(self.upconv1(dec_combined1))

        mu2, logvar2 = self.enc_combiner2(enc_z2, enc_z2).chunk(2, dim=1)
        delta_mu2, delta_logvar2 = self.enc_combiner2(enc_z2, dec_z2).chunk(2, dim=1)
        mu2_com = mu2 + delta_mu2
        logvar2_com = logvar2 + delta_logvar2
        sample2 = reparametrize(mu2_com, logvar2_com)
        dec_combined2 = self.dec_combiner2(sample2, dec_z2)
        dec_z3 = self.dec_res2(self.upconv2(dec_combined2))

        mu3, logvar3 = self.enc_combiner3(enc_z3, enc_z3).chunk(2, dim=1)
        delta_mu3, delta_logvar3 = self.enc_combiner3(enc_z3, dec_z3).chunk(2, dim=1)
        mu3_com = mu3 + delta_mu3
        logvar3_com = logvar3 + delta_logvar3
        sample3 = reparametrize(mu3_com, logvar3_com)
        dec_combined3 = self.dec_combiner3(sample3, dec_z3)
        dec_out = self.dec_res3(self.upconv3(dec_combined3))

        kl1 = kl(mu1, logvar1)
        kl2 = kl_rela(logvar2, delta_mu2, delta_logvar2)
        kl3 = kl_rela(logvar3, delta_mu3, delta_logvar3)
        kl_loss = kl1 + kl2 + kl3

        return dec_out, kl_loss

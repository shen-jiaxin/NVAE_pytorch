import torch
import torch.nn as nn
from nvae_utils import EncoderBlock, DecoderBlock, Combiner, reparametrize, kl, kl_rela, channels_cal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mseloss = nn.MSELoss(reduction='sum')


class NvaeModel(nn.Module):
    def __init__(self,
                 num_in_channel,  # the number of channels of input images
                 num_in_size,  # size of input images
                 batch_size,
                 latent_z_size=(64, 4, 4),  # size: C x H x W, all shoud be power of 2
                 double_group=True,  # option for doubling groups of latent variables
                 num_residual_cell=1,  # the number of residual cells in each group
                 ):

        super(NvaeModel, self).__init__()

        self.batch_size = batch_size

        # the number of groups of latent variables and the number of channels of each group
        self.num_latent_groups, self.num_channels_list = channels_cal(num_in_channel, num_in_size, latent_z_size, double_group)

        # original noise variable
        self.h = nn.Parameter(torch.randn((batch_size, latent_z_size[0], latent_z_size[1], latent_z_size[2])).cuda())

        # module lists for blocks (encoder and decoder) and combiners (encoder and decoder)
        self.encoder_block_list = []
        self.decoder_block_list = []
        self.dec_mu_logvar_combiner_list = []
        self.delta_combiner_list = []
        self.decoder_combiner_list = []
        for i in range(self.num_latent_groups):
            self.encoder_block_list.append(EncoderBlock(self.num_channels_list[i], self.num_channels_list[i+1], num_residual_cell))
            self.decoder_block_list.append(DecoderBlock(self.num_channels_list[-i - 1], self.num_channels_list[-i - 2], num_residual_cell))
            self.dec_mu_logvar_combiner_list.append(Combiner(self.num_channels_list[-i - 1], 'enc'))
            self.delta_combiner_list.append(Combiner(self.num_channels_list[-i - 1], 'enc'))
            self.decoder_combiner_list.append(Combiner(self.num_channels_list[-i - 1], 'dec'))

        self.encoder_block_list = nn.ModuleList(self.encoder_block_list)
        self.decoder_block_list = nn.ModuleList(self.decoder_block_list)
        self.encoder_combiner = Combiner(self.num_channels_list[i + 1], 'enc')
        self.dec_mu_logvar_combiner_list = nn.ModuleList(self.dec_mu_logvar_combiner_list)
        self.delta_combiner_list = nn.ModuleList(self.delta_combiner_list)
        self.decoder_combiner_list = nn.ModuleList(self.decoder_combiner_list)

    def encode(self, x):
        enc_z_list = []
        mu_logvar_list = []
        last_enc_z = x
        for i in range(self.num_latent_groups):
            enc_z = self.encoder_block_list[i](last_enc_z)
            enc_z_list.append(enc_z)
            last_enc_z = enc_z

        mu, logvar = self.encoder_combiner(enc_z, enc_z).chunk(2, dim=1)
        mu_logvar_list.append((mu, logvar))
        enc_z_list.reverse()
        mu1 = torch.reshape(mu_logvar_list[0][0], (self.batch_size, -1))
        logvar1 = torch.reshape(mu_logvar_list[0][1], (self.batch_size, -1))
        return enc_z_list, mu_logvar_list, mu1, logvar1

    def decode(self, z_list=[], enc_z_list=[], mu_logvar_list=[]):
        if enc_z_list:
            z_list = []
            z_list.append(reparametrize(mu_logvar_list[0][0], mu_logvar_list[0][1]))

        dec_mu_logvar_list = []
        delta_mu_logvar_list = []
        dec_combined = self.decoder_combiner_list[0](z_list[0], self.h)
        dec_z = self.decoder_block_list[0](dec_combined)

        for i in range(1, self.num_latent_groups):
            if enc_z_list:
                # get dec_mu and dec_logvar from higher level of groups
                dec_mu, dec_logvar = self.dec_mu_logvar_combiner_list[i](dec_z, dec_z).chunk(2, dim=1)

                # get delta_mu and delta_logvar from encoded_z and decoded_z
                delta_mu, delta_logvar = self.delta_combiner_list[i](enc_z_list[i], dec_z).chunk(2, dim=1)

                # get real mu and logvar for current group
                mu = dec_mu + delta_mu
                logvar = dec_mu + delta_logvar

                dec_mu_logvar_list.append((dec_mu, dec_logvar))
                delta_mu_logvar_list.append((delta_mu, delta_logvar))
                mu_logvar_list.append((mu, logvar))
                z_list.append(reparametrize(mu, logvar))

            dec_combined = self.decoder_combiner_list[i](z_list[i], dec_z)
            dec_z = self.decoder_block_list[i](dec_combined)

        recon = dec_z
        return recon, z_list, dec_mu_logvar_list, delta_mu_logvar_list, mu_logvar_list

    def forward(self, x):
        enc_z_list, enc_mu_logvar_list, _, _ = self.encode(x)
        recon, z_list, dec_mu_logvar_list, delta_mu_logvar_list, mu_logvar_list = self.decode([], enc_z_list, enc_mu_logvar_list)
        loss = self.loss_func(recon, x, dec_mu_logvar_list, delta_mu_logvar_list)
        return recon, mu_logvar_list, loss['loss']

    def loss_func(self, recon, original, mu_logvar_list, delta_mu_logvar_list):
        recon_loss = mseloss(recon, original)
        kl_loss = kl(mu_logvar_list[0][0], mu_logvar_list[0][1])
        for i in range(len(delta_mu_logvar_list)):
            kl_loss += kl_rela(mu_logvar_list[i][1], delta_mu_logvar_list[i][0], delta_mu_logvar_list[i][1])

        # total_loss is combination of reconstruction loss and KL-loss
        loss = 100 * recon_loss + kl_loss
        return {'loss': loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}

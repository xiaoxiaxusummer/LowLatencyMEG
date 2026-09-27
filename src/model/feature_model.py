import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from diffusers.models.autoencoders.vae import Decoder


class DynamicDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.feature_encoder = Net(args)

        self.vae_decoder = Decoder(
            in_channels        = 4,
            out_channels       = 3,
            up_block_types     = (
                        "UpDecoderBlock2D",
                        "UpDecoderBlock2D",
                        "UpDecoderBlock2D",
                        "UpDecoderBlock2D"),
            block_out_channels = (128, 256, 512, 512),
            layers_per_block   = 2,
            norm_num_groups    = 32,
            act_fn             = "silu",
        )

    def forward(self, x, epoch=600, channel_noise=0.1, train=False):
        latent_pred, KL = self.feature_encoder(x, epoch, channel_noise, train)
        output = self.vae_decoder(latent_pred)

        return output, KL, latent_pred



class gamma_layer(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(gamma_layer, self).__init__()
        self.H = nn.Parameter(torch.ones(output_channel, input_channel))
        self.b = nn.Parameter(torch.ones(output_channel))
        self.H.data.normal_(0, 0.1) # Initialize the parameters by narmal distribution parameterized by mean = 0 and std = 0.1
        self.b.data.normal_(0, 0.001)

    def forward(self, x):
        H = torch.abs(self.H)
        x = F.linear(x,H)
        return torch.tanh(x)

class gamma_function(nn.Module):

    def __init__(self, args):
        super(gamma_function, self).__init__()
        self.f1 = gamma_layer(1,32*32)
        self.f2 = gamma_layer(32*32,32*32)
        self.f3 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, stride=2, kernel_size=2, padding=0),
            nn.BatchNorm2d(16),
            nn.SELU(),
            nn.Conv2d(16, 16, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.SELU(),)
        self.f4 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, stride=2, kernel_size=2, padding=0),
            nn.BatchNorm2d(16),
            nn.SELU(),
            nn.Conv2d(16, 4, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.f1(x) # size: (1,) -> (32*32,)
        x = self.f2(x).reshape((1, 1, 32, 32)) # size: (32*32,) -> (32, 32)
        x = self.f3(x) # size: (1, 1, 32, 32) -> (1, 4, 64, 64)
        x = self.f4(x).reshape((1, 4*128*128))
        return x

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight

class Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.hidden_channel = 4*128*128
        c, w, h = args.latent_size # size of VAE encoding feature, default: (4, 128, 128)
        # self.encoder = nn.Sequential(
        #                 nn.Conv2d(4, 4, kernel_size = 1,stride = 1, padding = 0),
        #                 nn.SELU(),
        #                 nn.Conv2d(4, 4, kernel_size = 1,stride = 1, padding = 0),
        #                 nn.SELU()
        #             )      
        # self.decoder = nn.Sequential(
        #                 nn.Conv2d(4, 64, kernel_size = 1, stride = 1, padding = 0),
        #                 nn.SELU(), 
        #                 nn.Conv2d(64, 4, kernel_size = 1, stride = 1, padding = 0),
        #                 nn.SELU(), 
        #                 )
        self.post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)    
        self.selu = nn.SELU()
        self.gamma_mu = gamma_function(args).to(args.device, dtype=args.dtype)
        self.upper_tri_matrix = torch.triu(torch.ones((self.hidden_channel,self.hidden_channel))).to(args.device, dtype=args.dtype)


    def forward(self, x, epoch, channel_noise, train=False):
        x = torch.reshape(x,(x.size()[0],4*128*128)) #  (n_batch, dim)
        # x_norm2 = torch.norm(x,dim=1) # norm, size: (n_batch, dim) -> (n_batch, )
        # x = x.shape[-1] * (x.permute(1,0)/(x_norm2+1e-4)).permute(1,0) # x ./ x_norm2 (element-wise)
        mu = self.gamma_mu(channel_noise) # (dim, )
        encoded_feature = self.selu(x * mu) # equation (11), size: (n_batch, 64)
        # KL divergence
        KL = self.KL_log_uniform(channel_noise,torch.abs(encoded_feature)) # equation (6)
        # Gaussian channel noise
        x = encoded_feature + torch.randn_like(encoded_feature.detach()) * channel_noise # size: (n_batch, dim)
        if train:
            if epoch > 100:
                x = x * self.get_mask(mu,threshold = self.args.threshold).to(x.dtype) # enable pruning & padding after 60 training epoches
        else:
            x = x * self.get_mask(mu,threshold = self.args.threshold).to(x.dtype)
        output = self.post_quant_conv(x.reshape((x.shape[0], 4, 128, 128)))
        # output = self.decoder(x.reshape((x.shape[0], 4, 128, 128)))
        return output, KL * 0.1 / channel_noise

    def KL_log_uniform(self,channel_noise,encoded_feature):

        alpha = (channel_noise/(encoded_feature+1e-4))
        # print(alpha)
        k1 = 0.63576
        k2 = 1.8732
        k3 = 1.48695
        batch_size = alpha.size(0)
        KL_term = k1 * F.sigmoid(k2 + k3 * 2 * torch.log(alpha)) - 0.5 * F.softplus(-2 * torch.log(alpha)) - k1
        if torch.isnan(torch.sum(KL_term.detach())):
            print("NaN KL term")
        return - torch.sum(KL_term) / batch_size

    def get_mask(self, mu, threshold=None):
        if threshold is None:
            threshold = self.args.threshold
        alpha = mu.detach()
        hard_mask = (alpha > threshold).float()
        return hard_mask

    def get_mask_inference(self, channel_noise, threshold=None):
        if threshold is None:
            threshold = self.args.threshold
        mu = self.gamma_mu(channel_noise)
        # alpha = F.linear(mu, self.upper_tri_matrix)
        hard_mask = (mu > threshold).float()
        return hard_mask, mu
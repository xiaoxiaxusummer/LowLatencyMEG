import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.model import merge

# from src.utils.comm_utils import add_channel_noise
from diffusers.models.autoencoders.vae import Decoder



def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = init_generator(device)


class DynamicMergingDecoder(nn.Module):
    def __init__(self, args={"sx": 2, "sy": 2}, ori_feature_size=(128,128)):
        super().__init__()
        
        self.ori_size = ori_feature_size
        args.sx, args.sy = 2, 2
        self.args = args
        self.post_quant_conv = torch.nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)
        self.decoder = Decoder(
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


    def feature_merger(self, latent_code, num_pruning_neuron):
        args = self.args
        original_h, original_w = self.ori_size
        original_tokens = original_h * original_w # 原始token数目 = o_h * o_w
        downsample = int(math.ceil(math.sqrt(original_tokens // latent_code.shape[1]))) # 目前已下采样的级别
        w = int(math.ceil(original_w / downsample)) # 下采样的宽
        h = int(math.ceil(original_h / downsample)) # 下采样的高
        m, u = merge.bipartite_soft_matching_random2d(latent_code, w, h, args.sx, args.sy, num_pruning_neuron, no_rand=False, generator=generator)
        return m, u

    def forward(self, latents, pruning_ratio, channel_noise):
        KL = None
        # ================== 通道数*4, 长/2， 宽/2 ========================
        n_dim = latents.shape[-1]*latents.shape[-2]/4
        num_pruning_neuron = int(pruning_ratio *n_dim)
        latent_code = latents.permute(0,2,3,1).reshape((latents.shape[0],latents.shape[2]*latents.shape[3],latents.shape[1]))  
        latent_code = latent_code.reshape((latents.shape[0],int(latents.shape[2]*latents.shape[3]/4),int(latents.shape[1]*4))) # [n_batch, w*h/4, n_c*4]
        m, u = self.feature_merger(latent_code, num_pruning_neuron)
        merged_latents = m(latent_code)
        noised_latents = merged_latents + torch.randn_like(merged_latents.detach())*channel_noise
        unmerged_latents = u(noised_latents)
        unmerged_latents = unmerged_latents.reshape((latents.shape[0],latents.shape[2]*latents.shape[3],latents.shape[1]))
        unmerged_latents = unmerged_latents.reshape((latents.shape[0],latents.shape[2],latents.shape[3],latents.shape[1])).permute(0,3,1,2)

        x = self.post_quant_conv(unmerged_latents)
        y = self.decoder(x)
        return y, merged_latents.shape

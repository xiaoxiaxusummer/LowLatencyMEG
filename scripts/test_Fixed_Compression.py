import numpy as np
from dataclasses import asdict

import argparse
import torch
import torch.nn as nn


import torch, random, torchvision
import multiprocessing as mp

from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64, input_img_size=(3,1024,1024))

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import os, sys

current_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
project_dir = os.path.abspath(os.path.join(current_dir,"../"))
print(project_dir)
sys.path.append(project_dir)


from src.model.feature_merging import DynamicMergingDecoder
from src.utils.dataset_utils import get_dataset_loader
dataset = get_dataset_loader(split=False, multi_diffusion_step=True,small_dataset=True)

parser = argparse.ArgumentParser(description='Test Offline Distillation Performance')
parser.add_argument('--intermediate_dim', type=int, default=1024)
parser.add_argument('--latent_size', type=tuple, default=(4,128,128))
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=9e-3)
parser.add_argument('--threshold', type=float, default=1e-2)
parser.add_argument('--decay_step', type=int, default=60)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--weights', type=str)
parser.add_argument('--channel_noise', type=float, default = 0.1)
# parser.add_argument('--save_dir', type=str, default = project_dir + '/assets/dmvae_model/')
parser.add_argument('--save_img_dir', type=str, default = project_dir + '/results/samples/')
parser.add_argument('--pretrained_model_path', type=str, default = project_dir + '/assets/dmvae_model_diffstep_12/LAIONCOCO_ckpt_299_dmvae_model.pth')
# parser.add_argument('--vae_model_path', type=str, default = '')
parser.add_argument('--resume_training', type=bool, default = True)
parser.add_argument('--dtype', type=torch.dtype, default = torch.float16)
parser.add_argument('--test_dtype', type=torch.dtype, default = torch.float16)

args = parser.parse_args()
args.device = device


from dataclasses import dataclass
@dataclass
class TrainingConfig:
    image_size = 1024  # the generated image resolution
    batch_size = 1 # number of samples for each batch
    num_epochs = 200
    train_size = 20 # number of samples for each epoch
    test_size = 8
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    save_image_epochs = 10
    save_model_epochs = 20
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = ''  # the model saving path
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    n_training_step = int(train_size/batch_size)

def Calculate_FID(imgs_dist_true,imgs_dist):
    """
    Args:
        img_dist_true: perfect img tensors in int8
        imgs_dist: generated img tensors in int8
    Returns:
        FID score
    """

    fid.update(imgs_dist_true, real=True)
    fid.update(imgs_dist, real=False)
    fid_score = fid.compute()
    fid.reset()

    return fid_score

def Load_Imgs(img_files):
    if type(img_files) == str:
        imgs = torchvision.io.read_image(img_files).unsqueeze(0).repeat(4,1,1,1)
        return imgs
    else:
        imgs = []
        for p in img_files:
            imgs.append(torchvision.io.read_image(p))
        return torch.stack(imgs, 0)

def get_channel_noise(is_train=False, noise=0.1, device='cuda'):
    # Dynamic Channel Conditions
    if is_train:
        channel_noise = torch.rand(1)*0.27 + 0.05
    else:
        channel_noise = torch.FloatTensor([1]) * noise
    channel_noise = channel_noise.to(device)
    return channel_noise 

def test_MEG_fixed(noise=0.1):
    config = TrainingConfig()
    config.test_size = 20
    dmdecoder = DynamicMergingDecoder(args).to(args.device, args.test_dtype)
    model_data = torch.load(project_dir + f"/assets/dmvae_model/LAIONCOCO_ckpt_399_dmvae_model.pth")
    dmdecoder.load_state_dict(model_data['dmdecoder'])
    dmdecoder.requires_grad_(False)
    dmdecoder.eval()
    del model_data

    os.makedirs(args.save_img_dir + '/MEG/', exist_ok=True)
    os.makedirs(args.save_img_dir + '/CEG_Reference/', exist_ok=True)
    test_loader = dataset.select(np.arange(0,config.test_size,1))
    n_sample = int(config.test_size/config.batch_size)
    diffusion_steps = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    diff_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    merging_ratios = [0.1, 0.5, 0.8]
    with torch.no_grad():
        dmdecoder.eval()
        MSE_batch = np.zeros((len(diffusion_steps),len(merging_ratios)))
        latent_dim = np.zeros((len(diffusion_steps),len(merging_ratios)))
        FID_score = np.zeros((len(diffusion_steps),len(merging_ratios)))
        img_preds = torch.zeros((len(diffusion_steps),len(merging_ratios),3,1024,1024)).to(device, args.test_dtype)
        total = 0
        for sample_i in range(n_sample):
            shard = test_loader.shard(num_shards = n_sample, index = sample_i)
            target_images = (torch.tensor(shard["diffusion"][:,-1])).to(device, args.test_dtype)
            channel_noise = get_channel_noise(is_train=False, noise = noise).to(dtype=args.test_dtype)
            total += target_images.size(0)
            if sample_i == 7:
                print(f"============== Prompt: {shard['caption'][0]} ============")
                torchvision.utils.save_image(target_images, os.path.join(args.save_img_dir, f'CEG_Reference/sdxl_test_step_12.png'))
                imgs_perfect = Load_Imgs(os.path.join(args.save_img_dir, f'CEG_Reference/sdxl_test_step_12.png'))
            for d_i in range(len(diff_indexes)):
                for p_i,pruning_ratio in enumerate(merging_ratios):
                    latents = (torch.tensor(shard["latents"][:, diff_indexes[d_i]])).to(device, args.test_dtype)
                    img_pred, _, merged_feature_shape = dmdecoder(latents, pruning_ratio, channel_noise)
                    MSE_batch[d_i, p_i] += torch.nn.functional.mse_loss(img_pred.float(), target_images.float(), reduction="mean")
                    img_preds[d_i, p_i] = img_pred
                    aux_info = 64*64 - merged_feature_shape[1] # Length of auxiliary information, equal to the number of pruned dimension
                    latent_dim[d_i, p_i] += np.prod(merged_feature_shape) + aux_info
                    if sample_i == 7:
                        torchvision.utils.save_image(img_preds[d_i, p_i].unsqueeze(0), os.path.join(args.save_img_dir, f'MEG/sdxl_test_step_{diffusion_steps[d_i]}_ratio_{pruning_ratio}.png'))
                        imgs_MEG = Load_Imgs(os.path.join(args.save_img_dir, f'MEG/sdxl_test_step_{diffusion_steps[d_i]}_ratio_{pruning_ratio}.png'))
                        FID_score[d_i, p_i] = Calculate_FID(imgs_perfect, imgs_MEG)
        MSE_batch = MSE_batch/total
        latent_dim = latent_dim/total
        return MSE_batch, merging_ratios, latent_dim*16, FID_score

def test_VVAE(noise=0.1, threshold=None):
    args.threshold = threshold
    from src.model.feature_model import DynamicDecoder
    vvae_model = DynamicDecoder(args).to(device, args.test_dtype)
    config = TrainingConfig()
    config.test_size = 20
    model_data = torch.load(project_dir + f"/assets/vvae_model_diffstep/LAIONCOCO_ckpt_599_vvae_model.pth")
    vvae_model.load_state_dict(model_data['vvae_model'])
    vvae_model.requires_grad_(False)
    vvae_model.eval()
    del model_data

    os.makedirs(args.save_img_dir + '/MS_Pruning/', exist_ok=True)
    test_loader = dataset.select(np.arange(0,config.test_size,1))
    n_sample = int(config.test_size/config.batch_size)
    diffusion_steps = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    diff_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    with torch.no_grad():
        vvae_model.eval()
        MSE_batch = np.zeros((len(diffusion_steps),))
        FID_score = np.zeros((len(diffusion_steps),))
        img_preds = torch.zeros((len(diffusion_steps),3,1024,1024)).to(device, args.test_dtype)
        total = 0
        for sample_i in range(n_sample):
            shard = test_loader.shard(num_shards = n_sample, index = sample_i)
            target_images = (torch.tensor(shard["diffusion"][:,-1])).to(device, args.test_dtype)
            channel_noise = get_channel_noise(is_train=False, noise = noise).to(device, dtype=args.test_dtype)
            total += target_images.size(0)
            for d_i in range(len(diff_indexes)):
                latents = (torch.tensor(shard["latents"][:, diff_indexes[d_i]])).to(device, args.test_dtype)
                img_pred, KL, _  = vvae_model(latents, 600, channel_noise, train=False)
                MSE_batch[d_i] += torch.nn.functional.mse_loss(img_pred.float(), target_images.float(), reduction="mean")
                img_preds[d_i] = img_pred
                if sample_i == 7:
                    torchvision.utils.save_image(img_preds[d_i].unsqueeze(0), os.path.join(args.save_img_dir, f'MS_Pruning/sdxl_test_step_{diffusion_steps[d_i]}_threshold{threshold:.3f}.png'))
                    imgs_sample = Load_Imgs(os.path.join(args.save_img_dir, f'MS_Pruning/sdxl_test_step_{diffusion_steps[d_i]}_threshold{threshold:.3f}.png'))
                    imgs_perfect = Load_Imgs(os.path.join(args.save_img_dir, f'CEG_Reference/sdxl_test_step_12.png'))
                    FID_score[d_i] = Calculate_FID(imgs_perfect, imgs_sample)
        hard_mask, mu = vvae_model.feature_encoder.get_mask_inference(torch.FloatTensor([noise]).to(device, args.test_dtype),threshold=args.threshold)
        index = torch.nonzero(torch.lt(hard_mask,0.5)).squeeze(1)
        pruned_number = index.size()[0]
        # print(f"pruned dim: {pruned_number}, pruned ratio: {pruned_number/(4*128*128)}")
        activated_dim = 4*128*128 - pruned_number
        # print(f"activated_dim: {activated_dim}")
        trans_bit = activated_dim*16
        # print(f"transmission_bits: {trans_bit}")
        
        MSE_batch = MSE_batch/total
        # print(MSE_batch)
        return MSE_batch, pruned_number/(4*128*128), trans_bit, FID_score

def test_model_split(noise=0.1, finetuned=True):
    from diffusers import AutoencoderKL, AutoencoderTiny
    from diffusers.models.autoencoders.vae import Decoder
    decoder = Decoder(
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
    ).to(device, args.test_dtype)
    if not finetuned:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=args.dtype).to(device, args.dtype)
        decoder.load_state_dict(vae.decoder.state_dict())
    else:
        
        vae = torch.load(project_dir + f"/assets/decoder_model/LAIONCOCO_ckpt_299_model.pth")
        decoder.load_state_dict(vae["decoder"])
        del vae

    decoder.requires_grad_(False)
    decoder.eval()
    
    config = TrainingConfig()
    config.test_size = 20

    os.makedirs(args.save_img_dir + '/MS_Finetune/', exist_ok=True)
    test_loader = dataset.select(np.arange(0,config.test_size,1))
    n_sample = int(config.test_size/config.batch_size)
    diffusion_steps = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    diff_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    with torch.no_grad():
        MSE_batch = np.zeros((len(diffusion_steps),))
        FID_score = np.zeros((len(diffusion_steps),))
        img_preds = torch.zeros((len(diffusion_steps),3,1024,1024)).to(device, args.test_dtype)
        total = 0
        for sample_i in range(n_sample):
            shard = test_loader.shard(num_shards = n_sample, index = sample_i)
            target_images = (torch.tensor(shard["diffusion"][:,-1])).to(device, args.test_dtype)
            channel_noise = get_channel_noise(is_train=False, noise = noise).to(device, dtype=args.test_dtype)
            total += target_images.size(0)
            for d_i in range(len(diff_indexes)):
                latents = (torch.tensor(shard["latents"][:, diff_indexes[d_i]])).to(device, args.test_dtype)
                img_pred  = decoder(latents + channel_noise*torch.randn_like(latents))
                MSE_batch[d_i] += torch.nn.functional.mse_loss(img_pred.float(), target_images.float(), reduction="mean")
                img_preds[d_i] = img_pred
                if sample_i == 7:
                    torchvision.utils.save_image(img_preds[d_i].unsqueeze(0), os.path.join(args.save_img_dir, f'MS_Finetune/sdxl_test_step_{diffusion_steps[d_i]}.png'))
                    imgs_sample = Load_Imgs(os.path.join(args.save_img_dir, f'MS_Finetune/sdxl_test_step_{diffusion_steps[d_i]}.png'))
                    imgs_perfect = Load_Imgs(os.path.join(args.save_img_dir, f'CEG_Reference/sdxl_test_step_12.png'))
                    FID_score[d_i] = Calculate_FID(imgs_perfect, imgs_sample)
        
        MSE_batch = MSE_batch/total
        # print(MSE_batch)
        return MSE_batch, FID_score

if __name__=='__main__':

    os.makedirs(args.save_img_dir, exist_ok=True)
    print("--------------------- Perform MEG (fixed compression) -----------------")
    MSE_MEG, merging_ratio_MEG, trans_bit_MEG, FID_MEG = test_MEG_fixed()

    print("--------------------- Perform MS_Pruning (fixed compression) -----------------")
    MSE_MS_Pruning = np.zeros((9,9)) 
    FID_MS_Pruning = np.zeros((9,9))
    pruned_ratios_MS_Pruning, trans_bit_MS_Pruning  = [], []
    for i, th in enumerate([0.201, 0.25, 0.3163, 0.358, 0.4008, 0.446, 0.4834, 0.556, 0.618]):
        MSE, pruned_ratio, trans_bit, FID_score = test_VVAE(threshold=th)
        MSE_MS_Pruning[:,i] = MSE
        FID_MS_Pruning[:,i] = FID_score
        pruned_ratios_MS_Pruning.append(pruned_ratio)
        trans_bit_MS_Pruning.append(trans_bit)
    
    print("--------------------- Perform MS -----------------")
    MSE_MS, FID_MS = test_model_split(finetuned=False)
    
    print("--------------------- Perform MS_Finetune -----------------")
    MSE_MS_Finetune, FID_MS_Finetune = test_model_split()

    diffusion_steps = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    import matplotlib.pyplot as plt
    plt.plot(diffusion_steps, MSE_MS_Finetune, linestyle='--', color='firebrick', linewidth=1.5, marker='*', markersize=6, label=rf'MS-Finetune')  
    plt.plot(diffusion_steps, MSE_MS_Pruning[:,0], linestyle='--', color='seagreen', linewidth=1.5, marker='>', markersize=6,  label=rf'MS-Pruning, $\beta=${np.round(pruned_ratios_MS_Pruning[0],1)}')  
    plt.plot(diffusion_steps, MSE_MS_Pruning[:,1], linestyle='--', color='yellowgreen', linewidth=1.5, marker='x', markersize=6, label=rf'MS-Pruning, $\beta=${np.round(pruned_ratios_MS_Pruning[1],1)}')  
    plt.plot(diffusion_steps, MSE_MEG[:,0], linestyle='-', color='royalblue', linewidth=1.5, marker='o', markersize=6, label=rf'MEG (proposed), $\beta=${merging_ratio_MEG[0]}')  
    plt.plot(diffusion_steps, MSE_MEG[:,1], linestyle='-', color='deepskyblue', linewidth=1.5, marker='o', markersize=6, label=rf'MEG (proposed), $\beta=${merging_ratio_MEG[1]}') 
    plt.plot(diffusion_steps, MSE_MEG[:,2], linestyle='-', color='cyan', linewidth=1.5, marker='o', markersize=6, label=rf'MEG (proposed), $\beta=${merging_ratio_MEG[2]}')  
    plt.xlabel("Number of denoising steps")
    plt.ylabel("MSE of the reconstructed images")
    plt.ylim(0.000, 0.035)
    plt.legend()
    plt.savefig(f"{args.save_img_dir}/MSE_diff.png", format="png", bbox_inches='tight')
    plt.savefig(f"{args.save_img_dir}/MSE_diff.pdf", format="pdf", bbox_inches='tight')
    plt.close()

    
    print(f"============== MEG FID ============")
    print(FID_MEG)
    print(f"============== MS-Pruning FID ============")
    print(FID_MS_Pruning)
    print(f"============== MS-Finetune FID ============")
    print(FID_MS_Finetune)
    print(f"============== MS-Finetune FID ============")
    print(FID_MS)

    transmission_rate = 1.6647
    print("===================MEG Transmission latency (for 2 images): =================")
    print(trans_bit_MEG*2/10**6/transmission_rate)
    print("===================MS-Pruning Transmission latency (for 2 images): =================")
    print(np.array(trans_bit_MS_Pruning)*2/10**6/transmission_rate)
    print("===================MS-Finetune Transmission latency (for 2 images): =================")
    print(128*128*4*16*2/10**6/transmission_rate)
    print("===================Cloud Edge Generation Transmission latency (for 2 images): =================")
    print(1024*1024*3*16*2/10**6/transmission_rate)

    
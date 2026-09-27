import os, sys

current_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
project_dir = os.path.abspath(os.path.join(current_dir,"../"))
print(project_dir)
sys.path.append(project_dir)


"""Load the backbone model"""
from src.pipeline_MEG_SDXL import StableDiffusionXLMEGPipeline
from src.model.scheduler_perflow import PeRFlowScheduler

import random
import numpy as np

import torch, torchvision
torch.cuda.set_device(0)

import argparse
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
parser.add_argument('--save_img_dir', type=str, default = project_dir + '/results/samples/')
parser.add_argument('--pretrained_model_path', type=str, default = project_dir + '/assets/dmvae_model_diffstep_12/LAIONCOCO_ckpt_299_dmvae_model.pth')
parser.add_argument('--resume_training', type=bool, default = True)
parser.add_argument('--dtype', type=torch.dtype, default = torch.float16)

args = parser.parse_args()
args.device = 'cuda'


def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

pipe_backbone = StableDiffusionXLMEGPipeline.from_pretrained("xiaoxiaxu/LowLatencyMEG_Backbone", args=args, torch_dtype=torch.float16, use_safetensors=True)
pipe_backbone.scheduler = PeRFlowScheduler.from_config(pipe_backbone.scheduler.config, prediction_type="ddim_eps", num_time_windows=4)
cfg_sdxl = {
            'num_img': 4,
            'cfg_scale': 2.5,
            'prompt_pre': "photorealistic, uhd, high resolution, high quality, highly detailed; realistic photo, ", 
            'neg_prompt': "distorted, blur, low-quality, haze, out of focus",
            'height': 1024,
            'width': 1024,
            'latent_dim': 128,
            'output_type': 'pt',
}

import time 

pipe_backbone.to(args.device, torch.float32)
pipe_backbone.MEG_decoder.to(args.device, torch.float16)
pipe_backbone.MS_pruning_decoder.to(args.device, torch.float16)
pipe_backbone.MS_finetune_decoder.to(args.device, torch.float16)
prompts= ["Two people standing in the snow with skis"]
inference_time = []
with torch.no_grad():
    for i in range(2):
        setup_seed(seed=42)
        start_time = time.time()
        samples = pipe_backbone(
            prompt              = [cfg_sdxl['prompt_pre']+p for p in prompts], 
            negative_prompt     = [cfg_sdxl['neg_prompt']] * len(prompts),
            height              = 1024,
            width               = 1024,
            num_inference_steps = 8, 
            guidance_scale      = cfg_sdxl['cfg_scale'],
            output_type         = 'pt',
            algorithm_type      = 'MEG', # ["ECG", "MEG", "MS", "MS_pruning","MS_finetune"]
            feature_merging_ratio = 0.4, 
            num_images_per_prompt = 4,
        ).images
        end_time = time.time()
        if i > 0:
            inference_time.append((end_time - start_time)/len(prompts))
    torchvision.utils.save_image(samples[0], 'test_pipeline_img.png')
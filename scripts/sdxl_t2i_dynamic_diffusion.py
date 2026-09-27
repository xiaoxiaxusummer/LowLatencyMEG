import random, argparse, os
from pathlib import Path
import numpy as np
import scipy.io as scio
import torch, torchvision

import sys
sys.path.append('/homes/xx623/Workspace/AccMEG/main/')

from src.pipeline_sdxl_tome_timing import StableDiffusionXLToMePipeline
from src.scheduler_perflow import PeRFlowScheduler
import tomesd

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

torch.cuda.empty_cache()

for tomeratio in [0.5]:
    merging_type = f'pr_sdxl_ddiff_comm/pruning_{tomeratio}'
    Path("demo/"+merging_type).mkdir(parents=True, exist_ok=True) 

    diffusion_steps = np.linspace(start=4, num=12, stop=15, endpoint=True).astype(int)
    e2e_delays = np.zeros_like(diffusion_steps).astype(float)
    denoiser_delays = np.zeros_like(diffusion_steps).astype(float)

    cfg_scale_list = [2.5]  # suggest values [1.5, 2.0, 2.5]
    num_img = 2
    seed = 42
    prompts_list = [
        ["photorealistic, uhd, high resolution, high quality, highly detailed; masterpiece, A Ragdoll cat drinking tea in one teacup held in its paw, wearing a red jumper, and reading one book on the table", 
                    "distorted, blur, low-quality, haze, out of focus",],
        ["photorealistic, uhd, high resolution, high quality, highly detailed; RAW photo, a handsome man, wearing a black coat, outside, closeup face",
                    "distorted, blur, low-quality, haze, out of focus",],
        ["photorealistic, uhd, high resolution, high quality, highly detailed; masterpiece, A closeup face photo of girl, wearing a rain coat, in the street, heavy rain, bokeh,",
                    "distorted, blur, low-quality, haze, out of focus",],
        ["photorealistic, uhd, high resolution, high quality, highly detailed; RAW photo, a red luxury car, studio light",
                    "distorted, blur, low-quality, haze, out of focus",],
        ["photorealistic, uhd, high resolution, high quality, highly detailed; masterpiece, A beautiful cat bask in the sun",
            "distorted, blur, low-quality, haze, out of focus",], 
    ]
    n_cfg_scale, n_prompt = len(cfg_scale_list), len(prompts_list)

    for (d_i, num_diffusion_step) in enumerate(diffusion_steps):
        pipe = StableDiffusionXLToMePipeline.from_pretrained("hansyan/perflow-sdxl-dreamshaper", torch_dtype=torch.float16, use_safetensors=True, variant="v0-fix", force_download=True)
        pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="ddim_eps", num_time_windows=4)
        pipe.to("cuda", torch.float16)
        if tomeratio>0:
            tomesd.apply_patch(pipe, ratio=tomeratio, max_downsample=8)

        e2e_delay, denoiser_delay = 0, 0

        for cfg_scale in cfg_scale_list:
            for i, prompts in enumerate(prompts_list):
                setup_seed(seed)
                prompt, neg_prompt = prompts[0], prompts[1]
                samples = pipe(
                    prompt              = [prompt] * num_img, 
                    negative_prompt     = [neg_prompt] * num_img,
                    height              = 1024,
                    width               = 1024,
                    num_inference_steps = num_diffusion_step, 
                    # pruning_ratio       = 0.4,
                    guidance_scale      = cfg_scale,
                    output_type         = 'pt',
                ).images
                e2e_delay += samples[1] 
                denoiser_delay += samples[2]
                cfg_int = int(cfg_scale); cfg_float = int(cfg_scale*10 - cfg_int*10)
                save_name = f'step_{num_diffusion_step}_txt{i+1}_cfg{cfg_int}-{cfg_float}_ratio_{tomeratio}.png'
                torchvision.utils.save_image(torchvision.utils.make_grid(samples[0], nrow = num_img), os.path.join("demo", merging_type, save_name))
            
        e2e_delays[d_i] = e2e_delay/n_cfg_scale/n_prompt
        denoiser_delays[d_i] = denoiser_delay/n_cfg_scale/n_prompt
        
        scio.savemat(os.path.join("demo", merging_type, f'delays_tome_{tomeratio}_cfg{cfg_int}-{cfg_float}.mat'), {'e2e_delays':e2e_delays, 'denoiser_delays':denoiser_delays, 'diffusion_steps':diffusion_steps})

    print("e2e_delays:")
    print(e2e_delays)

    print("denoiser_delays:")
    print(denoiser_delays)


import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
project_dir = os.path.abspath(os.path.join(current_dir,"../"))
print(project_dir)
sys.path.append(project_dir+'/src/')
from diffusers import AutoPipelineForText2Image

import torch
import random
import numpy as np
import time

torch.cuda.empty_cache()
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


torch.cuda.set_device(0)

setup_seed(20)

def SDXL_Generation(prompt="photorealistic, uhd, high resolution, high quality, highly detailed, masterpiece, Two people standing in the snow with skis", save_dir=None):
    # Load SDXL-Turbo model
    model_id = "stabilityai/sdxl-turbo"
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")

    # If CUDA is enabled
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device, torch.float16)

    # Generate
    negative_prompt = "low quality, blurry"
    num_inference_steps=8
    inference_time = []
    for i in range(3):
        start_time = time.time()
        image = pipe(prompt, negative_prompt=negative_prompt, 
                     width = 1024, height = 1024,
                     num_inference_steps=num_inference_steps, guidance_scale=0.0).images[0]
        end_time = time.time()
        if i > 0:
            inference_time.append(end_time - start_time)
    
    print(f"Computational time of SDXL-Turbo to generate 1 image: {np.mean(inference_time):.4f} seconds")

    
    # Save
    image.save(save_dir + f"/SDXL_Turbo_{num_inference_steps}_step.png")


if __name__=='__main__':
    SDXL_Generation(save_dir='Workspace/AccMEG/results/samples')
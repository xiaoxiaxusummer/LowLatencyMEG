import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
project_dir = os.path.abspath(os.path.join(current_dir,"../../"))
sys.path.append(project_dir)

SAVE_DIR = os.path.join(os.path.abspath(os.path.join(project_dir,"../")), 'assets/laion-coco-aesthetic/')


from datasets import load_dataset, load_from_disk, Array4D, Array3D

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import requests
from datasets.utils.file_utils import get_datasets_user_agent

from PIL import Image
from torchvision import transforms

import validators
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib

from datasets import load_dataset, concatenate_datasets
from datasets import Image as dImg

import torch, random
from Workspace.AccMEG.git_LowLatencyMEG.src.pipeline_MEG_SDXL import StableDiffusionXLMEGPipelineOutput
from diffusers import AutoencoderKL
from src.model.scheduler_perflow import PeRFlowScheduler
import numpy as np


def get_dataset_loader(type="laion-coco-high-resolution", save_dir=SAVE_DIR, 
                       fetch_pipe_results=True, multi_diffusion_step = False, split=True, small_dataset=False):
    """
    @param downloading: whether to download the original dataset and save it by filtering
    @param fetch_pipe_results: whether use datasets with a 12-step SDXL results
    @param multi_diffusion_step: whether use datasets with SDXL generative results with denoising steps 4, 5, ..., 8
    @param split: whether split datasets into training and test data
    """
    if type=="laion-coco-high-resolution":
        # if downloading is True:
        #     dataset = load_dataset('guangyil/laion-coco-aesthetic', revision="refs/convert/parquet", split = 'train', 
        #                         cache_dir = os.path.join(project_dir,'assets/laion-coco-aesthetic'))
        #     sdxl_dataset = dataset.filter(lambda x: x['width'] >= 1024 and x['height']>=1024)
        #     sdxl_dataset = sdxl_dataset.filter(lambda x: validators.url(x['url']))
        #     num_shards = 500
        #     for shard_idx in range(6,num_shards):
        #         data_shard = sdxl_dataset.shard(num_shards=num_shards, index=shard_idx)
        #         data_shard = data_shard.map(invalid_images_as_none, batched=True, batch_size=100, num_proc=16)
        #         print(f"size of shard {shard_idx}: {len(data_shard)}")
        #         data_shard = data_shard.filter(lambda x: x['image'] is not None)
        #         print(f"size of shard {shard_idx}: {len(data_shard)} after filter")
        #         if len(data_shard) == 0: 
        #             print("no available samples in this shard")
        #             continue
        #         data_shard = data_shard.map(transform_img, batched=True, batch_size=100, num_proc=16)
        #         data_shard.save_to_disk(os.path.join(save_dir,f"parquet/{shard_idx:03d}.parquet"))
        dataset = None
        if fetch_pipe_results and multi_diffusion_step:
            num_files = 40 if small_dataset else 70
            for i in range(0, num_files):
                shard = load_from_disk(os.path.join(save_dir,f"ref_parquet_final/{i:03d}.parquet"))
                shard = shard.with_format("numpy", columns=["diffusion", "latents", "image"])
                dataset = shard if dataset is None else concatenate_datasets([dataset, shard])
        elif fetch_pipe_results:
            for i in range(0, 70):
                shard = load_from_disk(os.path.join(save_dir,f"ref_parquet_float32/{i:03d}.parquet"))
                shard = shard.with_format("numpy", columns=["diffusion", "latents", "image"])
                dataset = shard if dataset is None else concatenate_datasets([dataset, shard])
        else:
            for i in range(0, 10):
                shard = load_from_disk(os.path.join(save_dir,f"parquet/{i:03d}.parquet"))
                dataset = shard if dataset is None else concatenate_datasets([dataset, shard])
        if split:
            dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=666)
            return dataset["train"], dataset["test"]
        else:
            return dataset.shuffle(seed=666)
    else:
        return []

def invalid_images_as_none(batch):
    urls = batch["url"]
    images = [request_img(url) for url in urls]
    batch["ref_image"] = images
    return batch


def request_img(url):
    try:
        x = Image.open(requests.get(url, stream=True).raw)
    except:
        x = None
    return x

def transform_img(examples):
    
    Trans = transforms.Compose([
        transforms.Resize(1024), 
        transforms.CenterCrop((1024,1024)),
        transforms.ToTensor(),
    ])
    # examples["image"] = [Trans(image.convert("RGBA")) for image in examples["image"]]
    examples["ref_image"] = [Trans(Image.fromarray(image)) for image in examples["ref_image"]]
    return examples
import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
project_dir = os.path.abspath(os.path.join(current_dir,"../../"))
sys.path.append(project_dir)

SAVE_DIR = os.path.join(os.path.abspath(os.path.join(project_dir,"../")), 'main/assets/laion-coco-aesthetic/')


from datasets import load_dataset, load_from_disk, Array4D, Array3D

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datasets.utils.file_utils import get_datasets_user_agent

from PIL import Image
from torchvision import transforms

import io

from datasets import load_dataset, concatenate_datasets
from datasets import Image as dImg

import requests

def url_exists(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"文件下载成功: {save_path}")
    else:
        print(f"下载失败，状态码: {response.status_code}")



def download_all(parquet_start_idx, parquet_end_idx):
    for n_parquet in range(parquet_start_idx, parquet_end_idx):
        file_length = ["00011", "00010"]
        for i in range(11):
            url = f"{project_dir}/{n_parquet:03d}.parquet/data-{i:05d}-of-{file_length[0]}.arrow"
            if not url_exists(url):
                url = f"{project_dir}/{n_parquet:03d}.parquet/data-{i:05d}-of-{file_length[1]}.arrow"
            parquet_dir = os.path.join(save_dir, f"{n_parquet:03d}.parquet")
            os.makedirs(parquet_dir,exist_ok=True)
            save_path = os.path.join(parquet_dir, f"data-{i:05d}-of-00011.arrow")
            download_file(url, save_path)
            

        url = f"{project_dir}/{n_parquet:03d}.parquet/dataset_info.json"
        save_path = os.path.join(parquet_dir, "dataset_info.json")
        download_file(url, save_path)

        url = f"{project_dir}/{n_parquet:03d}.parquet/state.json"
        save_path = os.path.join(parquet_dir, "state.json")
        download_file(url, save_path)

save_dir = "Workspace/AccMEG/git_LowLatencyMEG/assets/laion-coco-aesthetic/multi_diffusion_step/"
project_dir = "https://huggingface.co/datasets/xiaoxiaxu/highresolution-laioncoco-aesthetic-MEG/tree/main/multi_diffusion_step"
parquet_start_idx = 0
parquet_end_idx = 2
download_all(parquet_start_idx, parquet_end_idx)
dataset=None
for i in range(0, 2):
    shard = load_from_disk(os.path.join(save_dir, f"{i:03d}.parquet"))
    dataset = shard if dataset is None else concatenate_datasets([dataset, shard])

dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=666)
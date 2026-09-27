import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import argparse
import copy
import numpy as np
import random, math
from pathlib import Path

from PIL import Image
import os, sys
from tqdm import tqdm
from accelerate import Accelerator



current_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
project_dir = os.path.abspath(os.path.join(current_dir,"../"))
sys.path.append(project_dir)


from diffusers.models.autoencoders.vae import Decoder
from diffusers.optimization import get_cosine_schedule_with_warmup

from src.utils.dataset_utils import get_dataset_loader
from src.model.feature_merging import DynamicMergingDecoder
from src.utils.comm_utils import add_channel_noise

from diffusers import AutoencoderKL, AutoencoderTiny
from src.model.scheduler_perflow import PeRFlowScheduler

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_channel_noise(is_train=False, noise=0.1, device='cuda'):
    # Dynamic Channel Conditions
    if is_train:
        channel_noise = torch.rand(1)*0.27 + 0.05
    else:
        channel_noise = torch.FloatTensor([1]) * noise
    channel_noise = channel_noise.to(device)
    return channel_noise      

cfg_sdxl = {
    'num_img': 1,
    'cfg_scale': 2.5,
    'tome_ratio': 0.5,
    'prompt_pre': "photorealistic, uhd, high resolution, high quality, highly detailed; realistic photo, ", 
    'neg_prompt': "distorted, blur, low-quality, haze, out of focus",
    'hight': 1024,
    'weight': 1024,
    'num_diffusion_step': 8,
    'output_type': 'pt',
}

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training VVAE')
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
parser.add_argument('--save_dir', type=str, default = project_dir + '/assets/dmvae_model/')
parser.add_argument('--save_img_dir', type=str, default = 'demo/pr_sdxl_ddiff_comm/dmvae_model/')
parser.add_argument('--pretrained_model_path', type=str, default = project_dir + '/assets/dmvae_model_diffstep_12/LAIONCOCO_ckpt_299_dmvae_model.pth')
# parser.add_argument('--vae_model_path', type=str, default = '')
parser.add_argument('--resume_training', type=bool, default = True)
parser.add_argument('--dtype', type=torch.dtype, default = torch.float32)
parser.add_argument('--test_dtype', type=torch.dtype, default = torch.float16)

args = parser.parse_args()
args.device = device
Path(args.save_dir).mkdir(parents=True, exist_ok=True) 
Path(args.save_img_dir).mkdir(parents=True, exist_ok=True) 

# ==========================     Dataset Loading      ==========================
train_dataset, test_dataset = get_dataset_loader(multi_diffusion_step=True)


# ========================== configure training and optimizer ========================

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
    output_dir = args.save_dir  # the model saving path
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    n_training_step = int(train_size/batch_size)

config = TrainingConfig()

# ========================== load pretrained vae decoder ========================

dmdecoder = DynamicMergingDecoder(args).to(device, args.dtype)
MSE_results = []


if not args.resume_training:
    start_epoch=0
    if not args.pretrained_model_path == '':
        decoder_pretrained = torch.load(args.pretrained_model_path)["dmdecoder"]
        dmdecoder.load_state_dict(decoder_pretrained)
        del decoder_pretrained
    else:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=args.dtype).to(device, args.dtype)
        dmdecoder.post_quant_conv.load_state_dict(vae.post_quant_conv.state_dict())
        dmdecoder.decoder.load_state_dict(vae.decoder.state_dict())
        del vae
else:
    start_epoch = 200
    data = torch.load(args.save_dir + f"/LAIONCOCO_ckpt_{start_epoch-1}_dmvae_model.pth")
    dmdecoder.load_state_dict(data['dmdecoder'])
    # lr_scheduler.load_state_dict(data['lr_scheduler'])
    MSE_results = data['MSE_results']
    del data


def get_lr_scheduler(start_steps, warmup_steps, total_training_steps, opt):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer= opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    for i in range(start_steps):
        scheduler.step()
    return scheduler


# --------------- for the first 300 epochs ---------------
# optimizer = torch.optim.AdamW(dmdecoder.parameters(), lr=config.learning_rate)
# lr_scheduler = get_lr_scheduler(start_epoch*config.n_training_step,  config.lr_warmup_steps, 
#                                 config.num_epochs*config.n_training_step,  optimizer)



# --------------- for 300-400 finetuning epochs ---------------
config.num_epochs = 400
optimizer = torch.optim.AdamW(dmdecoder.parameters(), lr = 5e-5)
lr_scheduler = get_lr_scheduler(0,  config.n_training_step*50, 150*config.n_training_step,  optimizer)

torch.cuda.empty_cache()



def train(config, dmdecoder, optimizer, train_dataset, lr_scheduler):
    global_step = start_epoch * config.train_size/config.batch_size
    test_MSE =  torch.inf
    pruned_dim = 0

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    dmdecoder, optimizer, train_dataset, lr_scheduler = accelerator.prepare(
        dmdecoder, optimizer, train_dataset, lr_scheduler
    )

    
    for epoch in range(start_epoch, config.num_epochs):
        dmdecoder.requires_grad_(True)
        dmdecoder.train()
        train_dataloader = train_dataset.select(np.random.randint(0,len(train_dataset),size=config.train_size)).shuffle()
        n_step = int(config.train_size/config.batch_size)
        progress_bar = tqdm(total=n_step, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        pruning_ratio = torch.rand((n_step,)).to(device, args.dtype)
        diff_step = np.random.randint(low=0, high=8, size=(n_step,))
        for step_i in range(n_step):
            shard = train_dataloader.shard(num_shards = n_step, index = step_i)
            # prompts = shard["caption"] # ref_image = shard["image"].to(device)
            sdxl_image = (torch.tensor(shard['diffusion'][:, -1])).to(device, args.dtype)
            latents = (torch.tensor(shard['latents'][:, diff_step[step_i]])).to(device, args.dtype)
            channel_noise = get_channel_noise(is_train=True)
            with accelerator.accumulate(dmdecoder):
                # Predict img
                img_pred, KL, _ = dmdecoder(latents, pruning_ratio[step_i], channel_noise)
                loss = torch.nn.functional.mse_loss(img_pred,sdxl_image,reduction="mean") 
                if torch.isnan(loss):
                    raise Exception("NaN value")
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(dmdecoder.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            MSE_batch, test_pruning_ratios = test(epoch, noise = 0.1)
            MSE_results.append(MSE_batch)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                torch.save({"dmdecoder": copy.deepcopy(dmdecoder.state_dict()), "MSE_results": MSE_results, "lr_scheduler": lr_scheduler.state_dict(), 
                            "epoch":epoch, "config":config, "args": args}, args.save_dir + f"/LAIONCOCO_ckpt_{epoch}_dmvae_model.pth")

            print('Test MSE:',MSE_batch[:, 0], 'Pruned ratio:', test_pruning_ratios)

        if epoch > 180:
            if np.mean(MSE_batch) < np.mean(test_MSE):
                torch.save({"dmdecoder": copy.deepcopy(dmdecoder.state_dict()), "MSE_results": MSE_results, 
                            'epoch':epoch, "config":config, "args": args}, 
                           args.save_dir + f"/LAIONCOCO_best_ckpt_dmvae_model.pth")
    print('Best Accuracy:',test_MSE,'Intermediate Dim:',args.intermediate_dim,'Beta:',args.beta)
    torch.save({"dmdecoder": copy.deepcopy(dmdecoder.state_dict()), "MSE_results": MSE_results, 'epoch':epoch, "config":config, "args": args}, 
               args.save_dir + f"/LAIONCOCO_final_ckpt_dmvae_model.pth")

def test(epoch, noise=0.1):
    test_loader = test_dataset.select(np.arange(0,config.test_size,1)).shuffle()
    n_step = int(config.test_size/config.batch_size)
    pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    diffusion_steps = [4, 12]
    diff_indexes = [0, 8]
    with torch.no_grad():
        dmdecoder.eval()
        MSE_batch = np.zeros((len(pruning_ratios),len(diffusion_steps,)))
        total = 0
        for step_i in range(n_step):
            shard = test_loader.shard(num_shards = n_step, index = step_i)
            # prompts = shard["caption"] # images = shard['image'].to(device)
            sdxl_images = (torch.tensor(shard["diffusion"][:,-1])).to(device, args.dtype)
            channel_noise = get_channel_noise(is_train=False, noise = noise).to(dtype=args.dtype)
            total += sdxl_images.size(0)
            img_preds = []
            for p_i,pruning_ratio in enumerate(pruning_ratios):
                for d_i in range(len(diff_indexes)):
                    latents = (torch.tensor(shard["latents"][:, diff_indexes[d_i]])).to(device, args.dtype)
                    img_pred, KL, _ = dmdecoder(latents, pruning_ratio, channel_noise)
                    MSE_batch[p_i, d_i] += torch.nn.functional.mse_loss(img_pred.float(), sdxl_images.float(), reduction="mean")
                    if d_i == 0:
                        img_preds.append(img_pred)
            if step_i == 0 and (((epoch+1) % config.save_image_epochs == 0) or (epoch+1==config.num_epochs) or epoch == 0):
                torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat([sdxl_images, torch.cat(img_preds,dim=0)],dim=0), nrow = 5), os.path.join(args.save_img_dir, f'sdxl_epoch{epoch}_step_{diffusion_steps[d_i]}.png'))
        return MSE_batch / total, pruning_ratios





if __name__=='__main__':
    seed_torch(0)
    if args.test == 0:
        train(config, dmdecoder, optimizer, train_dataset, lr_scheduler)
    else:
        dmdecoder.load_state_dict(torch.load(args.save_dir + f"/LAIONCOCO_best_ckpt_model.pth")['dmdecoder']).to(device, args.test_dtype)
        MSE_batch = 0
        t = 20
        for i in range (t):
            MSE, pruned_number = test(0, dmdecoder, args.channel_noise)
            MSE_batch += MSE
        print('Noise level:',args.channel_noise, 'Test Accuracy:', MSE_batch/t, 'Pruned dim:', pruned_number, 'Activated dim:', args.intermediate_dim - pruned_number)


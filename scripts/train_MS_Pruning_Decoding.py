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
from tqdm import tqdm
from accelerate import Accelerator


import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
project_dir = os.path.abspath(os.path.join(current_dir,"../"))
sys.path.append(project_dir)


from src.diffusers.models.autoencoders.vae import Decoder
from src.diffusers.optimization import get_cosine_schedule_with_warmup

from src.utils.dataset_utils import get_dataset_loader
from src.model.feature_model import DynamicDecoder

from src.pipeline_sdxl_tome_Fencoder import StableDiffusionXLToMePipeline
from src.diffusers import AutoencoderKL, AutoencoderTiny
from src.scheduler_perflow import PeRFlowScheduler

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

def get_lr_scheduler(start_steps, warmup_steps, total_training_steps, opt):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer= opt,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    for i in range(start_steps):
        scheduler.step()
    return scheduler

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

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training VVAE')
parser.add_argument('--intermediate_dim', type=int, default=128*128*4)
parser.add_argument('--latent_size', type=tuple, default=(4,128,128))
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--beta', type=float, default= 8e-6) # default=9e-3)
parser.add_argument('--threshold', type=float, default=2e-1) # default=1e-2
parser.add_argument('--decay_step', type=int, default=60)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--weights', type=str)
parser.add_argument('--channel_noise', type=float, default = 0.1)
parser.add_argument('--save_dir', type=str, default = project_dir + '/assets/vvae_model_diffstep/')
parser.add_argument('--save_img_dir', type=str, default = 'demo/pr_sdxl_ddiff_comm/vvae_model/')
# parser.add_argument('--vae_model_path', type=str, default = project_dir + '/assets/decoder_model/LAIONCOCO_ckpt_299_model.pth')
parser.add_argument('--vae_model_path', type=str, default = '')
parser.add_argument('--resume_training', type=bool, default = True)
parser.add_argument('--dtype', type=torch.dtype, default = torch.float32)

args = parser.parse_args()
args.device = device
Path(args.save_dir).mkdir(parents=True, exist_ok=True) 
Path(args.save_img_dir).mkdir(parents=True, exist_ok=True) 

# ==========================     Dataset Loading      ==========================
train_dataset, test_dataset = get_dataset_loader()


# ========================== configure training and optimizer ========================

from dataclasses import dataclass
@dataclass
class TrainingConfig:
    image_size = 1024  # the generated image resolution
    batch_size = 1 # number of samples for each batch
    train_size = 20 # number of samples for each epoch
    num_epochs_frozen = 10 # 150
    num_epochs = 600
    test_size = 8
    gradient_accumulation_steps = 1
    learning_rate_frozen = 1e-4
    learning_rate = 1e-4
    lr_warmup_steps = 2000
    lr_warmup_steps_frozen = 50
    save_image_epochs = 20
    save_model_epochs = 50
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = args.save_dir  # the model saving path
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    start_KL_epoch = 150
    n_training_step = int(train_size/batch_size) # number of training steps in each batch

config = TrainingConfig()

# ========================== load pretrained vae decoder ========================
vvae_model = DynamicDecoder(args).to(device, args.dtype)
start_epoch=0
MSE_results = []
if not args.resume_training:
    if not args.vae_model_path == '':
        decoder_pretrained = torch.load(args.vae_model_path)["decoder"]
        vvae_model.vae_decoder.load_state_dict(decoder_pretrained)
        del decoder_pretrained
    else:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=args.dtype).to(device, args.dtype)
        vvae_model.vae_decoder.load_state_dict(vae.decoder.state_dict())
        vvae_model.feature_encoder.post_quant_conv.load_state_dict(vae.post_quant_conv.state_dict())
        del vae
else:
    start_epoch = 400
    data = torch.load(args.save_dir + f"/LAIONCOCO_ckpt_{start_epoch-1}_vvae_model.pth")
    vvae_model.load_state_dict(data['vvae_model'])
    # lr_scheduler_frozen.load_state_dict(data['lr_scheduler_frozen'])
    # lr_scheduler.load_state_dict(data['lr_scheduler'])
    MSE_results = data['MSE_results']
    del data
    print(f"restart training from epoch{start_epoch}")

#  ----------------- during the first 400 epoch ---------------------
# optimizer_frozen = torch.optim.AdamW(vvae_model.feature_encoder.parameters(), lr=config.learning_rate_frozen)
# lr_scheduler_frozen = get_lr_scheduler(start_steps = start_epoch*config.n_training_step, warmup_steps=config.lr_warmup_steps_frozen, 
#                                        total_training_steps = config.n_training_step * config.num_epochs_frozen, opt=optimizer_frozen)

# optimizer = torch.optim.AdamW(vvae_model.parameters(), lr=config.learning_rate)
# lr_scheduler = get_lr_scheduler(start_steps = start_epoch*config.n_training_step, warmup_steps=config.lr_warmup_steps, 
#                                 total_training_steps = config.n_training_step * (config.num_epochs-config.num_epochs_frozen), opt=optimizer)

#  ----------------- during epoch 400-600 and 600 - 1000 for fine-tuning---------------------
optimizer_frozen, lr_scheduler_frozen = None, None
optimizer = torch.optim.AdamW(vvae_model.parameters(), lr = 1e-4)
lr_scheduler = get_lr_scheduler(start_steps = config.n_training_step * 0, warmup_steps = config.n_training_step * 50, 
                                total_training_steps = config.n_training_step * 150, opt=optimizer)

torch.cuda.empty_cache()


def train(config, vvae_model, optimizer_frozen, optimizer, train_dataset, lr_scheduler_frozen, lr_scheduler):
    test_MSE =  torch.inf
    pruned_dim = 0
    global_step = start_epoch * config.train_size/config.batch_size
    #  ======================= Train Feature Encoder and Frozen VAE Decoder =====================
    if start_epoch < config.num_epochs_frozen:
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

        vvae_model, optimizer_frozen, train_dataset, lr_scheduler_frozen = accelerator.prepare(
            vvae_model, optimizer_frozen, train_dataset, lr_scheduler_frozen
        )
        for epoch in range(start_epoch, config.num_epochs_frozen):
            vvae_model.feature_encoder.requires_grad_(True)
            vvae_model.feature_encoder.train()
            vvae_model.vae_decoder.requires_grad_(False)
            vvae_model.vae_decoder.eval()
            train_dataloader = train_dataset.select(np.random.randint(0,len(train_dataset),size=config.train_size)).shuffle()
            progress_bar = tqdm(total=config.n_training_step, disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            for step_i in range(config.n_training_step):
                shard = train_dataloader.shard(num_shards = config.n_training_step, index = step_i)
                prompts = shard["caption"] # ref_image = shard["image"].to(device)
                sdxl_image = torch.tensor(shard["diffusion"]).to(device, args.dtype)
                latents = torch.tensor(shard["latents"]).to(device, args.dtype)
                channel_noise = get_channel_noise(is_train=True)
                with accelerator.accumulate(vvae_model):
                    img_pred, KL, latent_pred  = vvae_model(latents, epoch, channel_noise, train=True)
                    loss = cal_loss(epoch, img_pred, sdxl_image, latent_pred, latents, KL)
                    if torch.isnan(loss):
                        raise Exception("NaN value")
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(vvae_model.feature_encoder.parameters(), 100.0)
                    optimizer_frozen.step()
                    lr_scheduler_frozen.step()
                    optimizer_frozen.zero_grad()
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler_frozen.get_last_lr()[0], "step": global_step}
                progress_bar.update(1)
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
            if accelerator.is_main_process:
                MSE_batch, pruned_number = test(epoch,noise = 0.1)
                MSE_results.append(MSE_batch)
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    torch.save({"vvae_model": copy.deepcopy(vvae_model.state_dict()), 
                                "lr_scheduler_frozen": lr_scheduler.state_dict(), "lr_scheduler": lr_scheduler.state_dict(), 
                                "MSE_results": MSE_results, "epoch":epoch, "config":config, "args": args}, args.save_dir + f"/LAIONCOCO_ckpt_{epoch}_vvae_model.pth")
            print('Test MSE:', MSE_batch, 'Pruned dim:',pruned_number,'Activated dim:', args.intermediate_dim - pruned_number)
        
        del accelerator
    
    #  ======================= Train Feature Encoder and Frozen VAE Decoder =====================
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    accelerator.init_trackers("train_example")
    vvae_model, optimizer, train_dataset, lr_scheduler = accelerator.prepare(
        vvae_model, optimizer, train_dataset, lr_scheduler
    )
    for epoch in range(max(start_epoch,config.num_epochs_frozen), config.num_epochs):
        vvae_model.requires_grad_(True)
        vvae_model.train()
        train_dataloader = train_dataset.select(np.random.randint(0,len(train_dataset),size=config.train_size)).shuffle()
        # n_step = int(config.train_size/config.batch_size)
        progress_bar = tqdm(total=config.n_training_step, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step_i in range(config.n_training_step):
            shard = train_dataloader.shard(num_shards = config.n_training_step, index = step_i)
            prompts = shard["caption"] # ref_image = shard["image"].to(device)
            sdxl_image = torch.tensor(shard["diffusion"]).to(device, args.dtype)
            latents = torch.tensor(shard["latents"]).to(device, args.dtype)
            channel_noise = get_channel_noise(is_train=True)
            with accelerator.accumulate(vvae_model):
                img_pred, KL, latent_pred  = vvae_model(latents, epoch, channel_noise, train=True)
                loss = cal_loss(epoch, img_pred, sdxl_image, latent_pred, latents, KL)
                if torch.isnan(loss):
                    raise Exception("NaN value")
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(vvae_model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        if accelerator.is_main_process:
            MSE_batch, pruned_number = test(epoch,noise = 0.1)
            MSE_results.append(MSE_batch)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1 or epoch==0:
                torch.save({"vvae_model": copy.deepcopy(vvae_model.state_dict()), 
                            "lr_scheduler_frozen": lr_scheduler.state_dict(), "lr_scheduler": lr_scheduler.state_dict(), 
                            "MSE_results": MSE_results, "epoch":epoch, "config":config, "args": args}, args.save_dir + f"/LAIONCOCO_ckpt_{epoch}_vvae_model.pth")
        print('Test MSE:', MSE_batch,'Pruned dim:',pruned_number, 'Activated dim:', args.intermediate_dim - pruned_number)

        if epoch > 180:
            if (MSE_batch < test_MSE and pruned_number == pruned_dim) or pruned_number > pruned_dim:
                test_MSE = MSE_batch
                pruned_dim = pruned_number
                print('Best ckpt:',test_MSE,'pruned_number:',pruned_dim,'beta:',args.beta)
                torch.save({"vvae_model": copy.deepcopy(vvae_model.state_dict()), 
                            'epoch':epoch, "config":config, "args": args}, args.save_dir + f"/LAIONCOCO_best_ckpt_vvae_model.pth")
    
    print('Best Accuracy:',test_MSE,'Intermediate Dim:',args.intermediate_dim,'Beta:',args.beta)
    torch.save({"vvae_model": copy.deepcopy(vvae_model.state_dict()), 
                'epoch':epoch, "config":config, "args": args}, args.save_dir + f"/LAIONCOCO_final_ckpt_vvae_model.pth")

def test(epoch, noise=0.1):
    test_loader = test_dataset.select(np.arange(0,config.test_size,1)).shuffle()
    n_step = int(config.test_size/config.batch_size)
    with torch.no_grad():
        vvae_model.eval()
        MSE_batch = 0
        total = 0
        for step_i in range(n_step):
            shard = test_loader.shard(num_shards = 20, index = step_i)
            prompts = shard["caption"] # images = shard['image'].to(device)
            sdxl_image = torch.tensor(shard["diffusion"]).to(device, args.dtype)
            latents = torch.tensor(shard["latents"]).to(device, args.dtype)
            channel_noise = get_channel_noise(is_train=False, noise = noise)
            img_pred, KL, _ = vvae_model(latents, epoch, channel_noise, train=False)
            total += latents.size(0)
            MSE_batch += torch.nn.functional.mse_loss(img_pred.float(), sdxl_image.float(), reduction="mean")
            if step_i == 0 and ((epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1):
                torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat([sdxl_image, img_pred],dim=0), nrow = 2), os.path.join(args.save_img_dir, f'sdxl_epoch{epoch}_step_{step_i}.png'))
        hard_mask, mu = vvae_model.feature_encoder.get_mask_inference(torch.FloatTensor([noise]).to(device))
        index = torch.nonzero(torch.lt(hard_mask,0.5)).squeeze(1)
        pruned_number = index.size()[0]
        return MSE_batch / total, pruned_number

def cal_loss(epoch, img_pred, sdxl_image, latent_pred, latents, KL):
    loss = torch.nn.functional.mse_loss(img_pred,sdxl_image,reduction="mean") * 10**2
    if epoch < config.num_epochs_frozen:
        loss += max(1-epoch/config.num_epochs,0) * torch.nn.functional.mse_loss(latent_pred,latents,reduction="mean") 
    if epoch > config.start_KL_epoch:
        anneal_ratio = min(1,(epoch - config.start_KL_epoch)/config.start_KL_epoch) 
        # print(f"KL: {KL}, ratio: {anneal_ratio}")
        loss += args.beta * KL.squeeze(0) * anneal_ratio
    return loss



if __name__=='__main__':
    seed_torch(0)
    if args.test == 0:
        train(config, vvae_model, optimizer_frozen, optimizer, train_dataset, lr_scheduler_frozen, lr_scheduler)
    else:
        # feature_model.load_state_dict(torch.load(args.weights)['feature_model'])
        vvae_model.load_state_dict(torch.load(args.save_dir + f"/LAIONCOCO_best_ckpt_model.pth")['vvae_model'])
        MSE_batch = 0
        t = 20
        for i in range (t):
            MSE, pruned_number = test(0,args.channel_noise)
            MSE_batch += MSE
        #print('Noise level:',args.channel_noise,'Test Accuracy:',accuracy/t,'Activated dim:', args.intermediate_dim - pruned_number)
        print('Noise level:',args.channel_noise, 'Test Accuracy:', MSE_batch/t, 'Pruned dim:', pruned_number, 'Activated dim:', args.intermediate_dim - pruned_number)


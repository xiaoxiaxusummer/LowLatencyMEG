import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import argparse
import copy
import numpy as np
import random
from pathlib import Path

from PIL import Image
import os, sys
from tqdm import tqdm
from accelerate import Accelerator

current_dir = os.path.abspath(os.path.dirname(__file__)) # 当前文件所在目录
project_dir = os.path.abspath(os.path.join(current_dir,"../"))
sys.path.append(project_dir)
from src.diffusers.optimization import get_cosine_schedule_with_warmup

from src.utils.dataset_utils import get_dataset_loader
from src.model.feature_model import Net

from src.pipeline_sdxl_tome_Fencoder import StableDiffusionXLToMePipeline
from src.diffusers import AutoencoderKL, AutoencoderTiny
from src.diffusers.models.autoencoders.vae import Decoder, DecoderTiny
# from src.diffusers.models.autoencoders.vae_float16 import Decoder
from src.scheduler_perflow import PeRFlowScheduler

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
parser.add_argument('--save_dir', type=str, default = project_dir + '/assets/decoder_model/')
parser.add_argument('--save_img_dir', type=str, default = project_dir + '/assets/ref_img/')
parser.add_argument('--vae_model_path', type=str, default = project_dir + '/assets/sdxl-vae--fp-16-fix/')
parser.add_argument('--resume_vae', type=bool, default = False)
parser.add_argument('--dtype', type=torch.dtype, default = torch.float32)

args = parser.parse_args()
args.device = device
Path(args.save_dir).mkdir(parents=True, exist_ok=True) 
Path(args.save_img_dir).mkdir(parents=True, exist_ok=True) 

# ==========================     Dataset Loading      ==========================
train_dataset, test_dataset = get_dataset_loader()
test_img_path = f'demo/pr_sdxl_ddiff_comm/decoder_model_finetune/'
Path(test_img_path).mkdir(parents=True, exist_ok=True) 

# ========================== configure training and optimizer ========================
from dataclasses import dataclass
@dataclass
class TrainingConfig:
    image_size = 1024  # the generated image resolution
    batch_size = 1 # number of samples for each batch
    num_epochs = 300
    train_size = 20 # number of samples for each epoch
    test_size = 8
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = project_dir + '/assets/decoder_model/'  # the model saving path
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()

# ========================== load pretrained vae decoder ========================
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
).to(device, args.dtype)
if not args.resume_vae:
    start_epoch = 0
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=args.dtype).to(device, args.dtype)
    decoder.load_state_dict(vae.decoder.state_dict())
    del vae
else:
    start_poch = 100
    data = torch.load(args.save_dir + f"/LAIONCOCO_ckpt_{start_poch-1}_model.pth")
    decoder.load_state_dict(data['decoder'].state_dict())
# vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype = args.dtype)
# decoder = DecoderTiny(
#     in_channels        = 4,
#     out_channels       = 3,
#     num_blocks         = (3,3,3,1,),
#     block_out_channels = (64,64,64,64),
#     upsampling_scaling_factor = 2,
#     act_fn             = "relu",
# ).to(device)
# pretrained_decoder = copy.deepcopy(decoder)
# pretrained_decoder.requires_grad_(False)

torch.cuda.empty_cache()


optimizer = torch.optim.AdamW(decoder.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(int(config.train_size/config.batch_size) * config.num_epochs),
)


def add_channel_noise(latents, train=False, noise = 0.1):
    if train:
        channel_noise = torch.rand(1)*0.27 + 0.05
    else:
        # noise = 0.1 
        channel_noise = torch.FloatTensor([1]) * noise
    return torch.randn_like(latents) * channel_noise.to(latents.device, latents.dtype)


def train(config, decoder, optimizer, train_dataset, lr_scheduler):
    test_MSE =  torch.inf
    pruned_dim = 0
    saved_model = {}

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
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    decoder, optimizer, train_dataset, lr_scheduler = accelerator.prepare(
        decoder, optimizer, train_dataset, lr_scheduler
    )

    global_step = 0
    decoder.requires_grad_(True)
    
    for epoch in range(start_epoch, config.num_epochs):
        # print('\nepoch:{}'.format(epoch))
        # train_dataloader = train_dataset.select(np.arange(0,config.train_size,1)).shuffle()
        train_dataloader = train_dataset.select(np.random.randint(0,len(train_dataset),size=config.train_size)).shuffle()
        n_step = int(config.train_size/config.batch_size)
        progress_bar = tqdm(total=n_step, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step_i in range(n_step):
            shard = train_dataloader.shard(num_shards = n_step, index = step_i)
            prompts = shard["caption"] # ref_image = shard["image"].to(device)
            sdxl_image = torch.tensor(shard["diffusion"]).to(device, args.dtype)
            latents = torch.tensor(shard["latents"]).to(device, args.dtype)
            channel_noise = add_channel_noise(latents, train=True, noise = 0.1)
            with accelerator.accumulate(decoder):
                # Predict img
                img_pred = decoder(latents + channel_noise)
                loss = torch.nn.functional.mse_loss(img_pred,sdxl_image,reduction="mean")
                if torch.isnan(loss):
                    raise Exception("NaN value")
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if step_i == 0:
                    torchvision.utils.save_image(torchvision.utils.make_grid(torch.cat([sdxl_image, img_pred],dim=0), nrow = 2), os.path.join(test_img_path, f'sdxl_epoch{epoch}_step_{step_i}.png'))
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            MSE_batch, pruned_number = test(epoch,noise = 0.1)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                torch.save({'decoder': copy.deepcopy(decoder.state_dict()), 'epoch':epoch, "config":config, "args": args}, 
                           args.save_dir + f"/LAIONCOCO_ckpt_{epoch}_model.pth")

        print('Test MSE:',MSE_batch,'Pruned dim:',pruned_number,'Activated dim:', args.intermediate_dim - pruned_number)

        if epoch > 180:
            if (MSE_batch < test_MSE and pruned_number == pruned_dim) or pruned_number > pruned_dim:
                test_MSE = MSE_batch
                pruned_dim = pruned_number
                print('Best ckpt:',test_MSE,'pruned_number:',pruned_dim,'beta:',args.beta)
                torch.save({'decoder': copy.deepcopy(decoder.state_dict()), 'epoch':epoch, "config":config, "args": args}, args.save_dir + f"/LAIONCOCO_best_ckpt_model.pth")
    print('Best Accuracy:',test_MSE,'Intermediate Dim:',args.intermediate_dim,'Beta:',args.beta)
    torch.save({'decoder': copy.deepcopy(decoder.state_dict()), 'epoch':epoch, "config":config, "args": args}, args.save_dir + f"/LAIONCOCO_final_ckpt_model.pth")

def test(epoch, noise=0.1):
    test_loader = test_dataset.select(np.arange(0,config.test_size,1)).shuffle()
    n_step = int(config.test_size/config.batch_size)
    with torch.no_grad():
        decoder.eval()
        MSE_batch = 0
        total = 0
        for step_i in range(n_step):
            shard = test_loader.shard(num_shards = 20, index = step_i)
            prompts = shard["caption"] # images = shard['image'].to(device)
            sdxl_image = torch.tensor(shard["diffusion"]).to(device, args.dtype)
            latents = torch.tensor(shard["latents"]).to(device, args.dtype)
            channel_noise = add_channel_noise(latents, train=False, noise = noise)
            noised_latents = latents + channel_noise
            img_pred = decoder(noised_latents)
            total += latents.size(0)
            MSE_batch += torch.nn.functional.mse_loss(img_pred.float(), sdxl_image.float(), reduction="mean")
        # hard_mask, mu = feature_model.get_mask_inference(torch.FloatTensor([noise]).to(device))
        # index = torch.nonzero(torch.lt(hard_mask,0.5)).squeeze(1)
        # pruned_number = index.size()[0]
        pruned_number = 0
        return MSE_batch / total, pruned_number




if __name__=='__main__':
    seed_torch(0)
    if args.test == 0:
        train(config, decoder, optimizer, train_dataset, lr_scheduler)
    else:
        # feature_model.load_state_dict(torch.load(args.weights)['feature_model'])
        decoder.load_state_dict(torch.load(args.weights)['decoder'])
        MSE_batch = 0
        t = 20
        for i in range (t):
            MSE, pruned_number = test(0,args.channel_noise)
            MSE_batch += MSE
        #print('Noise level:',args.channel_noise,'Test Accuracy:',accuracy/t,'Activated dim:', args.intermediate_dim - pruned_number)
        print('Noise level:',args.channel_noise, 'Test Accuracy:', MSE_batch/t, 'Pruned dim:', pruned_number, 'Activated dim:', args.intermediate_dim - pruned_number)


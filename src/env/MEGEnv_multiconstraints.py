import numpy as np
import torch, torchvision, math

import gymnasium as gym
from gymnasium import spaces

from pathlib import Path
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

import sys, os
current_path = os.path.abspath(os.path.dirname(__file__))
project_dir= os.path.abspath(os.path.join(current_path,'../../../'))
print("current project_dir:", project_dir)
sys.path.append(project_dir)
sys.path.append(project_dir+"/src/fsrl/")

from gymnasium.spaces import Box



def setup_pipe_seed(seed):
    import random
    torch.cuda.empty_cache()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class MEGEnv_multiconstraints(gym.Env):
    def __init__(self, dataset = None, dmdecoder=None, KL_in_loss = True,
                 MEG_mode="ddiff_MEG", args = None):
        
        # self.save_path = save_path # file path of the generated images and the label images
        # Path(save_path).mkdir(parents=True, exist_ok=True) 
        
        self.args = args
        self.KL_in_loss = KL_in_loss

        # ============== Performance Metrics Modelling ===========
        import scipy
        param = scipy.io.loadmat(os.path.join(project_dir,"scripts/pr_sdxl_ddiff/tome_0.5_perlayer/fit_linear_delays_tome_0.5_cfg2-5.mat"),squeeze_me=True)
        param = param['para_dict']
        self.param = {
            'flops_unets_single_step': float(param['flops_unets_single_step']),
            'flops_text_encoder': float(param['flops_text_encoder']),
            'flops_vae': float(param['flops_vae']),
            'computing_efficiency': float(param['computing_efficiency'])/1000, # ms/TFLOPs -> s/TFLOPs
            'basic_delay': float(param['basic_delay'])/1000, # ms -> s
            'energy_efficiency': float(param['energy_efficiency']), # hJ/TFLOPs 
            'bias_energy': float(param['bias_energy']), # hJ
        }
        self.param['UE_computing_efficiency_ratio'] = 0.5
        self.param['UE_energy_efficiency_ratio'] = 0.8
        self.param["UE_computing_efficiency"] = self.param["computing_efficiency"]/self.param['UE_computing_efficiency_ratio']
        self.param["UE_energy_efficiency"] = self.param["energy_efficiency"]/self.param['UE_energy_efficiency_ratio']

        # ==================== Time Frame Model =================
        T = args.Tmax
        self.cost_limit = [x/(args.Tmax-1) for x in args.cost_limit]
        self.Tmax = T+1 # total number of time periods for MEG
        self.execution_time = np.zeros((T,)) # execution time of each MEG time period
        self.batch_size = 1  # The size of the image numbers per batch

        self.radius = [200,300] 
        self.bandwidth = 1 # [MHz]
        self.Pmax_dBm = args.Pmax_dBm # maximum transmission power [dBm]
        self.Pmax = 10**(self.Pmax_dBm/10)/10**3 #  maximum transmission power [W]

        self.dtype = args.test_dtype
        self.np_dtype = np.float32 if self.dtype==torch.float32 else np.float16

        self.feature_noise = torch.zeros((self.Tmax,))

        if args.noise is None:
            self.PSNR_dB = 0
            self.PSNR = 10**(self.PSNR_dB/10) # signal-to-noise ratio
            self.noise = self.Pmax/self.PSNR # noise power [W]
        else:
            self.noise = args.noise # [W] # default: 0.1 W
            self.PSNR = self.Pmax/self.noise
            self.PSNR_dB = 10*math.log10(self.PSNR) # signal-to-noise ratio [dB]
        self.noise_dB = 10*np.log10(self.noise) # noise power [dB]
        self.N0 = 10**(-94/10)/1000 # additive Gaussian noise -95 dBm 

        # ===================== loading dataset ======================
        self.dataset_name = "LAIONCOCO"
        self.dataset = dataset
        self.dataset_size = len(self.dataset)

        # ================ loading diffusion pipeline ===============
        self.ref_img_width, self.ref_img_height = 1024, 1024
        self.img_width = 1024
        self.img_height = 1024
        self.ori_latent_size = 4*128*128
        self.dmdecoder = dmdecoder
        self.MEG_mode = MEG_mode
        
        self.action_space = Box(0, 1, shape=(2,))
        self.observation_space = Box(low=np.array([0, 0, 0]), 
                                     high=np.array([+np.inf]*3), shape=(3,), dtype=np.float32)
        # channel[t], latency[t-1], energy_consumption[t-1]



    def reset(self, seed=None, options=None):
        
        if seed is not None:
            super().reset(seed=seed) # seed for self.np.random
        
        T = self.Tmax
        self.t = 0

        self.reward = np.zeros((T,))
        self.cost = np.zeros((T,2))
        self.latency = np.zeros((T,))
        self.energy_consumption = np.zeros((T,))

        # ================ Initialize task buffer ================
        self.fetch_new_task(self.batch_size) 
        # self.remaining_FLOPs_UE = np.zeros((T,)) # FLOPs at the UE
        self.served_task = {
            'sample_idx': [], 
            'tran_latency': [],
            'comp_ES_latency': [],
            'comp_UE_latency': [], 
            'tran_energy': [], 
            'comp_ES_energy': [], 
            'comp_UE_energy': [], 
            'img_pred': [], 
            'ref_image': [], 
            'diffusion': [], 
        }
        
        
        # ============== Initialize wireless communication channels ===============
    
        # Reset the Rayleigh fading channels
        self.distance = self.radius[0] + np.random.rand(T)*(self.radius[1] - self.radius[0]) # [100, self.raduis]
        self.path_loss_dB = -(35.3+37.6*np.log10(self.distance))
        self.path_loss = 10**(self.path_loss_dB/10)  
        self.channel_gain_to_noise_ratio = self.Pmax*self.path_loss/self.N0

        # sample the Gaussian random noise
        self.feature_noise = get_feature_noise(shape=(T,), noise = self.noise, device=self.args.device, dtype=self.args.test_dtype)

        self.tran_latency = np.zeros((T,)) # transmission latency [second]
        self.tran_rate = np.zeros((T,)) 
                
        # ====================== MEG Parameters =======================
        self.FID_score = np.zeros((T,))
        self.MSE = np.zeros((T,))
        self.KL = np.zeros((T,))
        self.num_diffusion_step = np.zeros((T,))
        self.pruning_ratio = np.zeros((T,))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def _get_obs(self):  
        # output: (channel_t, latency_t-1,  energy_consumptiong_t-1)
        t = self.t
        if t > 0:
            return np.array([self.channel_gain_to_noise_ratio[t], self.latency[t-1], self.energy_consumption[t-1]])
        else:
            return np.array([self.channel_gain_to_noise_ratio[t], 0, 0])

    def _get_info(self):
        return{"cost": self.cost[self.t], "latency": self.latency[self.t], "task_sample": self.task_sample, "KL": self.KL[self.t], "MSE": self.MSE[self.t], "energy_consumption": self.energy_consumption[self.t], "pruning_ratio": self.pruning_ratio[self.t], "diff_step": self.num_diffusion_step[self.t], "cost_limit": self.cost_limit}

    def step(self, action):
        action = np.clip(action, a_min=0, a_max =1)
        t = self.t
        task_sample = self.task_sample
        diff_ratio = action[0]
        pruning_ratio = action[1]
        num_diffusion_step = int((12-4)*diff_ratio)+4
        self.pruning_ratio[t] = pruning_ratio
        n_diff_idx = [4,5,6,7,8,9,10,11,12].index(num_diffusion_step) # get the index of the selected diffusion step
        self.num_diffusion_step[t] = num_diffusion_step

        # handle current task batch
        sample_idx = task_sample[0] 
        data_samples = self.dataset.select(sample_idx)
        latents = torch.tensor(data_samples["latents"][:,n_diff_idx]).to(device=self.args.device, dtype=self.args.test_dtype)
        targets = torch.tensor(data_samples["diffusion"][:,-1]).to(device=self.args.device, dtype=self.args.test_dtype)
        with torch.no_grad():
            img_preds, KL, pruned_latent_shape = self.dmdecoder(latents, pruning_ratio, self.feature_noise[t])
            # calculate MSE/FID score
            self.MSE[t] = float(torch.nn.functional.mse_loss(img_preds, targets, reduction="mean"))
            # self.FID_score[t] = FID_score(torch.tensor(self.served_task["img_pred"]), self.served_task['ref_image'])
            self.KL[t] = float(KL)
            if self.KL_in_loss:
                self.reward[t] = -np.log(self.MSE[t]) - self.args.beta_KL * self.KL[t]
            else:
                self.reward[t] = -np.log(self.MSE[t])
        # transmission metrics
        latent_dim = np.prod(np.array(pruned_latent_shape))
        tran_bit = (latent_dim + (self.ori_latent_size - latent_dim)*1/16) * int(str(self.dtype)[-2:])
        # tran_rate = self.bandwidth*np.log2(1+self.Pmax*self.channel[t]/self.noise)
        tran_rate = self.bandwidth*np.log2(1+self.channel_gain_to_noise_ratio[t])
        self.tran_latency[t] = tran_bit/10**6/tran_rate
        self.tran_rate[t]= tran_rate
        tran_energy = self.Pmax * self.tran_latency[t]

        
        # computation metrics
        img_ratio = self.img_height/self.ref_img_height * self.img_width/self.ref_img_width
        FLOPs_dyndiff_ES = (self.param["flops_unets_single_step"] * num_diffusion_step )* img_ratio + self.param["flops_text_encoder"]
        FLOPs_dyndiff_UE = self.param["flops_vae"] * img_ratio
        comp_ES_latency = FLOPs_dyndiff_ES*self.param["computing_efficiency"]
        comp_UE_latency = FLOPs_dyndiff_UE*self.param["UE_computing_efficiency"]
        comp_ES_energy = FLOPs_dyndiff_ES*self.param["energy_efficiency"] # hJ/TFLOPs
        comp_UE_energy = FLOPs_dyndiff_UE*self.param["UE_energy_efficiency"]  # hJ/TFLOPs

        if t == 0:
            self.execution_time[t] = comp_ES_latency + self.tran_latency[t]
        else:
            self.execution_time[t] = max(comp_ES_latency-self.tran_latency[t-1],0) + self.tran_latency[t]

        self.latency[t] = np.mean(self.tran_latency[t] + comp_ES_latency + comp_UE_latency + self.param["basic_delay"])
        self.energy_consumption[t] = (np.mean(comp_ES_energy +comp_UE_energy + tran_energy + self.param["bias_energy"]))/10 # kJ
        # update served task list
        self.update_served_task(task_sample, self.tran_latency[t], comp_ES_latency, comp_UE_latency, 
                             comp_ES_energy, tran_energy, comp_UE_energy, 
                             img_preds, data_samples["diffusion"][:,-1])
        self.cost[t] = [self.latency[t], self.energy_consumption[t]]
        
        info = self._get_info()

        if (t+2==self.Tmax):
            print(f"mean reward: {self.reward[0:t].mean()}, mean cost: {self.cost[0:t].mean()}, mean KL: {self.KL[0:t].mean()},  mean MSE: {self.MSE[0:t].mean()}, mean tran rate: {self.tran_rate[0:t].mean()},  mean tran latency: {self.tran_latency[0:t].mean()}, pruning ratio: {self.pruning_ratio[0:t].mean()}, diff step: {self.num_diffusion_step[0:t].mean()}")

        # update the time slot
        self.t += 1
        done = True if self.t >= self.Tmax else False
        terminated = done

        # fetch new task batch
        self.fetch_new_task(self.batch_size)

        obs_next = self._get_obs()

        return obs_next, self.reward[t], terminated, done, info


    def fetch_new_task(self, num_sample):
        """fetch one task batch sample from the task buffer in a first-in-first-out manner
        input: 
            - num_sample: number of required samples in each batch
        output:
            - task_sample: tuple(sample_idx, )
        """
        t = self.t
        if t > 0:
            self.task_sample = (
                (self.np_random.integers(0,self.dataset_size-1,size=(num_sample,))).tolist(),
                )
        else:
            self.task_sample = (
                    (self.np_random.integers(0,self.dataset_size-1,size=(num_sample,))).tolist(),
                )
        return self.task_sample

    def update_served_task(self, task_samples, tran_latency, comp_ES_latency, comp_UE_latency, 
                        tran_energy, comp_ES_energy, comp_UE_energy, 
                        img_pred, ori_img):
        """
        add one/multipe task sample(s) into the task buffer in a first-in-first-out manner
        """
        num_new_task = self.batch_size
        self.served_task["sample_idx"].extend(task_samples[0])
        self.served_task["tran_latency"].extend([tran_latency]*num_new_task)
        self.served_task["comp_ES_latency"].extend([comp_ES_latency]*num_new_task)
        self.served_task["comp_UE_latency"].extend([comp_UE_latency]*num_new_task)
        self.served_task["tran_energy"].extend([tran_energy]*num_new_task)
        self.served_task["comp_ES_energy"].extend([comp_ES_energy]*num_new_task)
        self.served_task["comp_UE_energy"].extend([comp_UE_energy]*num_new_task)
        self.served_task["img_pred"].extend(img_pred)
        # self.served_task["ref_image"].extend(ref_img)
        self.served_task["diffusion"].extend(ori_img)



def FID_score(img_tensors, label_tensors):
    """
        calculate FID scores for two set of image tensors 
        input: 
        - img_tensors -> tuple(num_img, 3, width, height)
        - label_tensors -> tuple(num_img, 3, width, height)
        output:
        - FID score -> float
    """
    assert img_tensors.shape[1] == 3 and label_tensors.shape[1] == 3
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(label_tensors, real=True)
    fid.update(img_tensors, real=False)
    print(f"FID: {float(fid.compute())}")
    return float(fid.compute())

def get_label_img(sample_idx, label_path, dataset):
    img = []
    for data in dataset[sample_idx]:
        img_path = os.path.join(label_path, data[0])
        img.append(Image.open(img_path))
    return torch.tensor(img)

def get_feature_noise(shape=None, noise=0.1, device='cuda', dtype=torch.float32):
    # Dynamic Channel Conditions
    # if is_train:
    #     feature_noise = torch.rand(shape)*0.27 + 0.05
    # else:
    #     feature_noise = torch.FloatTensor(shape) * noise
    feature_noise = torch.FloatTensor(torch.ones(shape)) * noise
    return feature_noise.to(device, dtype)

if __name__ == "__main__":

    from fsrl.config.cvpo_cfg_MEG import TrainCfg, TrainingConfig
    from src.model.feature_merging import DynamicMergingDecoder

    from src.utils.dataset_utils import get_dataset_loader
    dataset = get_dataset_loader(split=False, multi_diffusion_step=True,small_dataset=True)

    TORCH_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16
    }
    
    args = TrainCfg()
    # set environment parameters
    args.cost_limit = [3*(args.Tmax-1), 1.7*(args.Tmax-1)]
    args.test_dtype = TORCH_DTYPES[args.test_dtype]
    args.Pmax_dBm = 30
    args.noise = 0.1

    # load the frozen vae decoder
    dmdecoder_config = TrainingConfig()
    torch.cuda.set_device(0)
    args.device = 'cuda'
    dmdecoder = DynamicMergingDecoder(args).to(args.device, args.test_dtype)
    if args.device == 'cpu':
        model_data = torch.load( os.path.abspath(project_dir+"/../") + f"/main/assets/dmvae_model/LAIONCOCO_ckpt_399_dmvae_model.pth", map_location=torch.device('cpu'))
    else:
        model_data = torch.load( os.path.abspath(project_dir+"/../") + f"/main/assets/dmvae_model/LAIONCOCO_ckpt_399_dmvae_model.pth")
    dmdecoder.load_state_dict(model_data['dmdecoder'])
    dmdecoder.requires_grad_(False)
    dmdecoder.eval()
    del model_data
    del dmdecoder_config
    
    env = MEGEnv_multiconstraints(dataset=dataset, dmdecoder=dmdecoder, args=args, KL_in_loss=False)

    obs, info = env.reset()

    for t in range(args.Tmax):
        act = [0.5, 0.5]
        obs_next, reward, terminated, done, info = env.step(act)
        print(f"t: {t}, rew: {reward}, cost: {info['cost']}, MSE: {info['MSE']}, Energy: {info['energy_consumption']}, pr ratio:{env.pruning_ratio[t]}, diffusion step:{env.num_diffusion_step[t]}, obs: {obs_next}")

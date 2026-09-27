import os, sys

current_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.abspath(os.path.join(current_dir,"../../../"))
sys.path.append(project_dir)
sys.path.append(project_dir+"/src/MEG_RL/")


import pprint
from dataclasses import asdict


import gymnasium as gym
import src.env # Import related class to register gym

import pyrallis
import torch
import torch.nn as nn
from tianshou.data import VectorReplayBuffer
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb
from torch.distributions import Independent, Normal

import torch, random
import multiprocessing as mp

torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
   mp.set_start_method('spawn', force=True)
   print("spawned")
except RuntimeError:
   pass

from fsrl.config.cvpo_cfg_MEG import TrainCfg, TrainingConfig
from fsrl.data import FastCollector
from fsrl.policy import CVPO
from fsrl.trainer import OffpolicyTrainer
from fsrl.utils import TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.utils.net.continuous import DoubleCritic, SingleCritic
from src.model.feature_merging import DynamicMergingDecoder
from src.utils.dataset_utils import get_dataset_loader
dataset = get_dataset_loader(split=False, multi_diffusion_step=True,small_dataset=True)

TORCH_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16
}


def get_lr_scheduler(optimizer, initial_lr, min_lr, coeff=5):
    lambda1 = lambda epoch: max(0.99 ** (epoch*coeff), min_lr / initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch = -1)
    return scheduler

@pyrallis.wrap()
def train(args: TrainCfg, cost_limit: float, power: float):
    # set seed and computing
    seed_all(args.seed)
    torch.set_num_threads(args.thread)

    # set environment parameters
    args.cost_limit = [x*(args.Tmax-1) for x in cost_limit]
    args.Pmax_dBm = power    
    args.mstep_iter_num = 5
    args.estep_iter_num = 10
    args.update_per_step = 0.6
    args.sample_act_num = 32
    args.hidden_sizes = (128, 128)
    args.noise = 0.1

    cfg = asdict(args)
    default_cfg = asdict(args)
    args.test_dtype = TORCH_DTYPES[args.test_dtype]

    # setup logger
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(cost_limit)
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)
    
    # load the frozen vae decoder
    dmdecoder_config = TrainingConfig()
    dmdecoder = DynamicMergingDecoder(args).to(args.device, args.test_dtype)
    model_data = torch.load(os.path.abspath(os.path.join(project_dir,"../")) + f"/assets/dmvae_model/LAIONCOCO_ckpt_399_dmvae_model.pth")
    dmdecoder.load_state_dict(model_data['dmdecoder'])
    dmdecoder.requires_grad_(False)
    dmdecoder.eval()
    del model_data
    del dmdecoder_config

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    train_envs = worker([lambda: gym.make(args.task, dataset=dataset, dmdecoder=dmdecoder, args=args, KL_in_loss=False) for _ in range(training_num)])
    test_envs = worker([lambda: gym.make(args.task, dataset=dataset, dmdecoder=dmdecoder, args=args, KL_in_loss=False) for _ in range(args.testing_num)])

    # model
    env = gym.make(args.task, dataset=dataset, dmdecoder=dmdecoder, args=args, KL_in_loss=False) 
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    assert hasattr(
        env.spec, "max_episode_steps"
    ), "Please use an env wrapper to provide 'max_episode_steps' for CVPO"

    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net,
        action_shape,
        max_action=max_action,
        device=args.device,
        conditioned_sigma=args.conditioned_sigma,
        unbounded=args.unbounded
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    # actor_lr_scheduler = get_lr_scheduler(actor_optim, args.actor_lr, min_lr=6e-9, coeff=3)


    critics = []
    for i in range(len(args.cost_limit)+1):
        if args.double_critic:
            net1 = Net(
                state_shape,
                action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device
            )
            net2 = Net(
                state_shape,
                action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device
            )
            critics.append(DoubleCritic(net1, net2, device=args.device).to(args.device))
        else:
            net_c = Net(
                state_shape,
                action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device
            )
            critics.append(SingleCritic(net_c, device=args.device).to(args.device))

    critic_optim = torch.optim.Adam(
        nn.ModuleList(critics).parameters(), lr=args.critic_lr
    )
    # critics_lr_scheduler = get_lr_scheduler(critic_optim, args.critic_lr, min_lr=5e-8, coeff=3)

    if not args.conditioned_sigma:
        torch.nn.init.constant_(actor.sigma_param, -0.5)
    actor_critic = ActorCritic(actor, critics)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    if args.last_layer_scale:
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = CVPO(
        actor=actor,
        critics=critics,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        logger=logger,
        action_space=env.action_space,
        dist_fn=dist,
        max_episode_steps=env.spec.max_episode_steps,
        cost_limit=args.cost_limit,
        tau=args.tau,
        gamma=args.gamma,
        n_step=args.n_step,
        # E-step
        estep_iter_num=args.estep_iter_num,
        estep_kl=args.estep_kl,
        estep_dual_max=args.estep_dual_max,
        estep_dual_lr=args.estep_dual_lr,
        sample_act_num=args.sample_act_num,  # for continous action space
        # M-step
        mstep_iter_num=args.mstep_iter_num,
        mstep_kl_mu=args.mstep_kl_mu,
        mstep_kl_std=args.mstep_kl_std,
        mstep_dual_max=args.mstep_dual_max,
        mstep_dual_lr=args.mstep_dual_lr,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        lr_scheduler=None,
    )

        
    if args.resume == True:
        resumed_model = torch.load(args.resume_path)
        policy.load_state_dict(resumed_model["model"])


    # collector
    train_collector = FastCollector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=False,
    )
    test_collector = FastCollector(policy, test_envs)

    def stop_fn(reward, cost):
        return reward > args.reward_threshold and cost < args.cost_limit

    def checkpoint_fn():
        return {"model": policy.state_dict()}

    if args.save_ckpt:
        logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        cost_limit=args.cost_limit,
        step_per_epoch=args.step_per_epoch,
        update_per_step=args.update_per_step,
        episode_per_test=args.testing_num,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_model_interval=args.save_interval,
        verbose=args.verbose,
    )

    for epoch, epoch_stat, info in trainer:
        logger.store(tab="train", cost_limit=args.cost_limit)
        print(f"Epoch: {epoch}")
        print(info)

    if __name__ == "__main__":
        pprint.pprint(info)
        # Let's watch its performance!
        env = gym.make(args.task, dataset=dataset, dmdecoder=dmdecoder, args=args, KL_in_loss=False) 
        policy.eval()
        collector = FastCollector(policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        policy.train()
        collector = FastCollector(policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")



if __name__ == "__main__":
    energy_limit = 1.8
    for cost_l in [3]:
        for power in [30]:
            train(cost_limit = [cost_l, energy_limit], power = power)

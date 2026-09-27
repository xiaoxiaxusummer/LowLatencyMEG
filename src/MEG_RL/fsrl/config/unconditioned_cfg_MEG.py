from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class TrainCfg:
    # 手动添加如下参数
    Tmax: int = 64
    use_default_cfg: bool = True
    beta_MSE: float = 1
    beta_latency: float = 1
    beta_dB: float = float(np.log10(beta_latency/beta_MSE)*10)
    test_dtype: str = "float16"
    Pmax_dBm: float = 30
    noise: float = None
    SNR_dB: float = 10 # dB

    # general task params
    task: str = "MEG-Merging-KL"
    cost_limit: float = 1e5
    device: str = "cuda"
    thread: int = 1 # 4  # if use "cpu" to train
    seed: int = 10
    # CVPO arguments
    estep_iter_num: int = 1
    estep_kl: float = 0.02
    estep_dual_max: float = 20
    estep_dual_lr: float = 0.02
    sample_act_num: int = 16
    mstep_iter_num: int = 1
    mstep_kl_mu: float = 0.005
    mstep_kl_std: float = 0.0005
    mstep_dual_max: float = 0.5
    mstep_dual_lr: float = 0.1
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    gamma: float = 0.97
    n_step: int = 2
    tau: float = 0.05
    hidden_sizes: Tuple[int, ...] = (128, 128)
    double_critic: bool = False
    conditioned_sigma: bool = False # 原始值: True
    unbounded: bool = False
    last_layer_scale: bool = False
    # collecting params
    epoch: int = 100 # 200
    episode_per_collect: int = 4 # 10
    step_per_epoch: int = 256 # 10000
    update_per_step: float = 0.2 # 0.2
    buffer_size: int = 200000
    worker: str = "ShmemVectorEnv"
    training_num: int = 1 # 20
    testing_num: int = 1 # 2
    # general train params
    batch_size: int = 256 # 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = False
    render: bool = False
    # logger params
    logdir: str = "logs"
    project: str = "ddpg-MEG"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "cvpo"
    suffix: Optional[str] = ""


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
    output_dir = None  # the model saving path
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    n_training_step = int(train_size/batch_size)

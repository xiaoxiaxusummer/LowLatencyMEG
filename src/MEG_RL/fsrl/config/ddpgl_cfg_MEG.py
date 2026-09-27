from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainCfg:
    # 手动添加如下参数
    Tmax: int = 64
    use_default_cfg: bool = True
    beta_KL: float = 1e-6
    test_dtype: str = "float16"
    task_arrival_interval: int = 3
    Pmax_dBm: float = 30
    noise: float = 0.1
    
    # general task params
    task: str = "MEG-Merging-nq-new"
    cost_limit: float = Tmax * 3
    device: str = "cuda"
    thread: int = 1  # if use "cpu" to train
    seed: int = 10
    # algorithm params
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (128, 128)
    tau: float = 0.05
    exploration_noise: float = 0.1
    n_step: int = 2
    # Lagrangian specific arguments
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.97
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    epoch: int = 100
    episode_per_collect: int = 4
    step_per_epoch: int = 245
    update_per_step: float = 0.2
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 1
    testing_num: int = 1
    # general params
    batch_size: int = 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # logger params
    logdir: str = "logs"
    project: str = "MEG-constrained"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "ddpgl"
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

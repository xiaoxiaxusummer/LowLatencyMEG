import torch

def add_channel_noise(device, dtype, train=False, noise = 0.1):
    if train:
        channel_noise = torch.rand(1)*0.27 + 0.05
    else:
        channel_noise = torch.FloatTensor([1]) * noise
    return channel_noise.to(device, dtype)
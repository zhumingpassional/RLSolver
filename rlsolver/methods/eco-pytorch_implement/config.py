import torch

GPU_ID = 6

def calc_device(gpu_id: int):
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
DEVICE = calc_device(GPU_ID)
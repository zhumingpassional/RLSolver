from torch.func import vmap
import torch
import time

x = torch.rand(2,3,4,device='cuda')

def f(x):

    return x*2

vmaped_f = vmap(f,in_dims=0)

vvmaped_f = vmap(vmaped_f,in_dims=1)
print(vvmaped_f(x))

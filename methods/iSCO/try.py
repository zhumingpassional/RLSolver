from torch.func import vmap
import torch
import time

x = torch.rand(8000,500,device='cuda')

def f(x):
    for i in range(300000):
        x = torch.softmax(x,dim=-1)
    return x

vmaped_f = vmap(f,in_dims=0)

start_time = time.time()
f(x)
print(f'原生耗时{time.time()-start_time}')
start_time2 = time.time()
vmaped_f(x)
print(f'vmap耗时{time.time()-start_time2}')

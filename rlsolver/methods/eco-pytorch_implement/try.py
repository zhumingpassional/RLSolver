import numpy as np
import torch
import time


spins = 2*torch.randint(0,2,(20,))-1
matrix = 2*torch.randint(0,2,(200,200))-1
start = time.time()
for i in range(10000):
    b=matrix.fill_diagonal_(0)
print(time.time()-start)



# a=(1 / 4) * torch.sum(matrix * (1 - torch.outer(spins, spins)))
spins = spins.numpy()
matrix = matrix.numpy()
start = time.time()
for i in range(10000):
    np.fill_diagonal(matrix, 0)
print(time.time()-start)
# b=(1/4) * np.sum( np.multiply( matrix, 1 - np.outer(spins, spins) ) )
# print(a,b)
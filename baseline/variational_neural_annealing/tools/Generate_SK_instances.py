import numpy as np
import random
import os

"""
This is code is used to generate the random instances for the Sherrington-Kirkpatrick model from a given set of seeds
in a format that is suitable for submission to the spinglass server https://software.cs.uni-koeln.de/spinglass/
Code by Mohamed Hibat-Allah
"""

list_seeds = [i for i in range(1,40+1)]
## Only the 25 seeds: "1,2,3,5,6,7,8,9,13,16,18,19,21,22,23,25,27,30,31,32,34,35,38,39,40" have been successfully solved by the spin-glass server https://software.cs.uni-koeln.de/spinglass/
## Those 25 seeds are the ones we run with our variational classical annealing implementation in our paper.

N = 100

for seed in list_seeds:
	random.seed(seed)  # `python` built-in pseudo-random generator
	np.random.seed(seed)  # numpy pseudo-random generator

	Jz = np.random.normal(0,1/np.sqrt(N),size = (N,N))

	if not os.path.exists('./configs/'):
		os.mkdir('./configs')
	file = open("./configs/"+str(N)+"_SK_seed"+str(seed)+".txt", "w")

	#Print the couplings to submit to https://software.cs.uni-koeln.de/spinglass/
	for i in range(N):
		for j in range(i+1,N):
		    file.write(str(i+1) + " "+str(j+1)+" "+str(Jz[i,j])+"\n")
	file.close()
	print("seed ", seed, " finished.")

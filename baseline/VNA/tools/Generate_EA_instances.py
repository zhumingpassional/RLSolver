import numpy as np
import random
import os

"""
This is code is used to generate the random instances for the 2D Edward Anderson model from a given set of seeds
in a format that is suitable for submission to the spinglass server https://software.cs.uni-koeln.de/spinglass/
Code by Mohamed Hibat-Allah
"""

list_seeds = [i for i in range(1,25+1)]

Nx = 40
Ny = 40

for seed in list_seeds:
	random.seed(seed)  # `python` built-in pseudo-random generator
	np.random.seed(seed)  # numpy pseudo-random generator

	Jz = np.random.uniform(0,2, size = (Nx,Ny,2))-1

	if not os.path.exists('./configs/'):
		os.mkdir('./configs')
	file = open("configs/"+str(Nx)+"x"+str(Ny)+"_uniform_seed"+str(seed)+".txt", "w")

	#Print the couplings to submit to https://software.cs.uni-koeln.de/spinglass/
	for ny in range(Ny):
		for nx in range(Nx):
			if nx != Nx-1:
			    file.write(str(nx+ny*Nx+1) + " "+str(nx+1+ny*Nx+1)+" "+str(Jz[nx,ny,0])+"\n")
			if ny != Ny-1:
			    file.write(str(nx+ny*Nx+1) + " " +  str(nx+(ny+1)*Nx+1)+" "+str(Jz[nx,ny,1])+"\n")
	file.close()
	print("seed ", seed, " finished.")

import tqdm
import numpy as np
from scipy.stats import poisson
import torch
import matplotlib.pyplot as plt
import time
import array
import os

class Experiment:
    def __init__(self):
        self.chain_length = 50000
        self.init_temperature = 1.0
        self.final_temperature = 0.000001
        self.data_directory =r"../../data/syn_BA"
    
    def get_data_list_and_init_sample(self,model,sampler,filename):
        data_list,tensor_dict = model.get_data_list(filename)
        init_sample = sampler.get_init_sample(model)
        return data_list,tensor_dict,init_sample
    
    def write_result(self,result,energy,data_directory, file,running_duration,model):
        output_filename = os.path.join(r"../../result",("result_"+file))
        with open(output_filename, 'w', encoding="UTF-8") as file:
            if running_duration is not None:
                file.write(f'// running_duration: {running_duration}\n')
            if energy is not None:
                file.write(f'// score: {energy}\n')
            for node in range(model.max_num_nodes):
                file.write(f'{node + 1} {int(result[node] + 1)}\n')




    def get_results(self,model,sampler):
        for file in os.listdir(self.data_directory):
            filename = os.path.join(self.data_directory, file)

            start_time = time.time()

            energy_list = []
            path_length_list = []
            sample_list = []
            acc_list = []
            sample_array = array.array("f")
            success_count = 0
            energy = 0
            result = 0
            data_list,tensor_dict,sample = self.get_data_list_and_init_sample(model,sampler,filename)
            temperature = model.init_temperature
            mu = 10
            for step in tqdm.tqdm(range(1,self.chain_length)):
                poisson_dist = poisson(mu)
                path_length = max(1,int(poisson_dist.rvs(size=1)))
                path_length_list.append(path_length)
                temperature = self.init_temperature * ( 1- step / self.chain_length)
                sample,new_energy,average_acc,success_count = sampler.step(path_length,tensor_dict,temperature,model,sample,success_count)
                if new_energy < energy:
                    energy = new_energy
                    result = sample
                average_acc = int(average_acc.cpu().item())
                mu = min(800,max(1,(mu + 0.001*(average_acc - 0.574))))
            energy = -(int(energy.cpu().item()))
            end_time = time.time()
            elapsed_time = end_time - start_time
            result = result.tolist()[0]


            self.write_result(result,energy,self.data_directory, file,elapsed_time,model)
            


            print(energy,success_count)
        # print(tensor_dict['sol'])
        # a = energy / tensor_dict['sol']
#g15 2500 2541 3048 g14 3051

def build_expirement():
    return Experiment()

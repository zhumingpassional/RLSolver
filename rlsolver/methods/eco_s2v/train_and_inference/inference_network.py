import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import time
import torch
from typing import List
import networkx as nx
from rlsolver.methods.eco_s2v.src.envs.inference_network_env import SpinSystemFactory
from rlsolver.methods.eco_s2v.util import eeco_test_network, load_graph_from_txt,load_graph_set_from_folder
from rlsolver.methods.eco_s2v.src.envs.eeco_util import (SetGraphGenerator,
                                                    RewardSignal, ExtraAction,
                                                    OptimisationTarget, SpinBasis,
                                                    DEFAULT_OBSERVABLES, Observable)
from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN
from rlsolver.methods.util_read_data import read_nxgraphs
from rlsolver.methods.util_result import write_graph_result
from rlsolver.methods.eco_s2v.config.config import *
import json


"""
逻辑是先读网络文件夹中的网络，再读图文件夹中的图，对一张图开n个环境,取最大值，结果文件要以网络文件命名，结果文件中的内容是图的名称.
在测试网络的过程中，为提高效率，图文件夹中只放一张图
"""
def run(graph_folder="../../data/syn_BA",
        network_folder=None,
        if_greedy=False,
        n_sims=1):
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))
    result_folder = os.path.dirname(network_folder) #保存结果的文件夹
    networks = os.listdir(network_folder)
    obj_vs_time = {}
    for network_name in networks:
        if network_name.endswith(".pth"):
            network_time = network_name.split("_")[-1].split(".")[0]
            network_save_path = os.path.join(network_folder, network_name)
            parts = network_save_path.split(".")
            network_result_save_path = parts[0] + ".json"
            print("Testing network: ", network_save_path)

            network_fn = MPNN
            network_args = {
                'n_layers': 3,
                'n_features': 64,
                'n_hid_readout': [],
                'tied_weights': False
            }
            network = network_fn(n_obs_in=7,
                        **network_args).to(INFERENCE_DEVICE)

            network.load_state_dict(torch.load(network_save_path, map_location=INFERENCE_DEVICE))
            for param in network.parameters():
                param.requires_grad = False
            network.eval()

            if ALG == Alg.eco or ALG == Alg.eeco:
                env_args = {'observables': DEFAULT_OBSERVABLES,
                            'reward_signal': RewardSignal.BLS,
                            'extra_action': ExtraAction.NONE,
                            'optimisation_target': OptimisationTarget.CUT,
                            'spin_basis': SpinBasis.BINARY,
                            'norm_rewards': True,
                            'memory_length': None,
                            'horizon_length': None,
                            'stag_punishment': None,
                            'basin_reward': 1. / NUM_TRAIN_NODES,
                            'reversible_spins': True,
                            'if_greedy':if_greedy,
                            'use_tensor_core': USE_TENSOR_CORE}

            step_factor = 2
            files = os.listdir(graph_folder)

            for prefix in INFERENCE_PREFIXES:
                graphs = []
                file_list = []
                for file in files:
                    if prefix in file:
                        file = os.path.join(graph_folder, file).replace("\\", "/")
                        file_list.append(file)
                        g = load_graph_from_txt(file)
                        g_array = nx.to_numpy_array(g)
                        g_tensor = torch.tensor(g_array, dtype=torch.float, device=INFERENCE_DEVICE)
                        start_time = time.time()
                        graphs.append(g_tensor)
                if len(graphs) > 0:
                    for i, graph_tensor in enumerate(graphs):
                        test_graph_generator = SetGraphGenerator(graph_tensor,device=INFERENCE_DEVICE)

                        test_env = SpinSystemFactory.get(test_graph_generator,
                                                    graph_tensor.shape[0] * step_factor,
                                                    **env_args,device = INFERENCE_DEVICE,
                                                        n_sims = n_sims,
                                                        )

                    start_time = time.time()
                    result,sol = eeco_test_network(network,test_env,USE_TENSOR_CORE,INFERENCE_DEVICE)
                    result['graph_name'] = files[i]
                    run_duration=time.time() - start_time
                    print(result['obj'],run_duration)
       
                    # write_graph_result(result['obj'], run_duration, sol.shape[0], ALG.value, ((sol+1)/2), file_list[i], plus1=True)  
                    if obj_vs_time.get(files[i]) is None:
                        obj_vs_time[files[i]] = {}
                    if network_time == "best":
                        network_time = "0"
                    obj_vs_time[files[i]][network_time] = result['obj']
                    print("Result: ", result['obj'])  
    with open(os.path.join(network_folder, "logger.json"), 'r') as f:
        data = json.load(f)
        data['obj_vs_time'] = {}
    with open(os.path.join(network_folder, "inference_logger.json"), 'w') as f:
        for filename, time_data in obj_vs_time.items():
            # 按照时间（键）进行排序，并生成新的字典
            sorted_time_data = {k: time_data[k] for k in sorted(time_data, key=int)}
            # 更新原字典
            data['obj_vs_time'][filename] = sorted_time_data
        json.dump(data, f, indent=4)
    print("Results saved to: ", os.path.join(network_folder, "inference_logger.json"))
if __name__ == "__main__":
    run(graph_folder=DATA_DIR, 
        network_folder=INFERENCE_NETWORK_DIR,
        if_greedy=False,
        n_sims=NUM_INFERENCE_SIMS)
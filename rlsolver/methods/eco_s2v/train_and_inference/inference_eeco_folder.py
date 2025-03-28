import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import time
import torch
import networkx as nx
import shutil
from rlsolver.methods.eco_s2v.src.envs.inference_network_env import SpinSystemFactory
from rlsolver.methods.eco_s2v.util import eeco_test_network, load_graph_from_txt
from rlsolver.methods.eco_s2v.src.envs.eeco_util import (SetGraphGenerator,
                                                         RewardSignal, ExtraAction,
                                                         OptimisationTarget, SpinBasis,
                                                         DEFAULT_OBSERVABLES)
from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN

from rlsolver.methods.util_result import write_graph_result
from rlsolver.methods.eco_s2v.config import *
import json


"""
逻辑是先读网络文件夹中的网络，再读图文件夹中的图，对一张图开n个环境,取最大值，结果文件要以网络文件命名，结果文件中的内容是图的名称.
在测试网络的过程中，为提高效率，图文件夹中只放一张图
"""
def run(graph_folder="../../data/syn_BA",
        network_folder=None,
        if_greedy=False,
        n_sims=1,
        mini_sims=10):  # 设置 mini_sims 以减少显存占用
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))
    network_result_save_path = network_folder + "/" +network_folder.split("/")[-1]+ ".json"
    networks = os.listdir(network_folder)
    obj_vs_time = {}
    data = {}
    best_network = [0,None]
    for network_name in networks:
        if network_name.endswith(".json"):
            with open(os.path.join(network_folder, network_name),"r") as f:
                data = json.load(f)
        if network_name.endswith(".pth"):
            network_time = network_name.split("_")[-1].split(".")[0]
            network_save_path = os.path.join(network_folder, network_name)
            print("Testing network: ", network_save_path)

            network_fn = MPNN
            network_args = {
                'n_layers': 3,
                'n_features': 64,
                'n_hid_readout': [],
                'tied_weights': False
            }
            network = network_fn(n_obs_in=7, **network_args).to(INFERENCE_DEVICE)

            network.load_state_dict(torch.load(network_save_path, map_location=INFERENCE_DEVICE))
            for param in network.parameters():
                param.requires_grad = False
            network.eval()

            if ALG == Alg.eco or ALG == Alg.eeco:
                env_args = {
                    'observables': DEFAULT_OBSERVABLES,
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
                    'if_greedy': if_greedy,
                    'use_tensor_core': USE_TENSOR_CORE_IN_INFERENCE
                }

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
                        graphs.append(g_tensor)

                if len(graphs) > 0:
                    for i, graph_tensor in enumerate(graphs):
                        test_graph_generator = SetGraphGenerator(graph_tensor, device=INFERENCE_DEVICE)

                        best_obj = float('-inf')
                        best_sol = None

                        num_batches = (n_sims + mini_sims - 1) // mini_sims  # 计算分批次数
                        for batch in range(num_batches):
                            current_mini_sims = min(mini_sims, n_sims - batch * mini_sims)  # 防止超出 n_sims

                            test_env = SpinSystemFactory.get(
                                test_graph_generator,
                                graph_tensor.shape[0] * step_factor,
                                **env_args,
                                device=INFERENCE_DEVICE,
                                n_sims=current_mini_sims,  # 只处理 mini_sims 个环境
                            )

                            start_time = time.time()
                            result, sol = eeco_test_network(network, test_env, USE_TENSOR_CORE_IN_INFERENCE, INFERENCE_DEVICE)

                            if result['obj'] > best_obj:  # 记录最佳结果
                                best_obj = result['obj']
                                best_sol = result['sol']
                        run_duration = time.time() - start_time
                        sol = (best_sol + 1)/2
                        write_graph_result(best_obj, run_duration, sol.shape[0], ALG.value, sol.to(torch.int), file_list[i], plus1=True)
                        if best_obj > best_network[0]:
                            best_network[0] = best_obj
                            best_network[1] = network_name
                        if obj_vs_time.get(files[i]) is None:
                            obj_vs_time[files[i]] = {}
                        if network_time == "best":
                            network_time = "0"
                        obj_vs_time[files[i]][network_time] = best_obj
                        print("Result: ", best_obj)
    if not data:
        data['n_smis'] = 1
    data['obj_vs_time'] = obj_vs_time
    with open(network_result_save_path,'w') as f:
        json.dump(data,f, indent=4)
        
    if best_network[1] is not None:
    # 复制最佳网络文件到目标文件夹
        best_network_path = os.path.join(network_folder, best_network[1])
        target_path = network_folder.replace("tmp/", "") + "best.pth"
        shutil.copy(best_network_path, target_path)  # 复制文件
if __name__ == "__main__":
    run(graph_folder=DATA_DIR,
        network_folder=NETWORK_FOLDER,
        if_greedy=False,
        n_sims=NUM_INFERENCE_SIMS,
        mini_sims=MINI_INFERENCE_SIMS)
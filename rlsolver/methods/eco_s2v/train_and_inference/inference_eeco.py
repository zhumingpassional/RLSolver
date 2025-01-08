import os
import time
import torch
from typing import List
import networkx as nx
import rlsolver.methods.eco_s2v.src.envs.core as ising_env
from rlsolver.methods.eco_s2v.util import eeco_test_network, load_graph_from_txt,load_graph_set_from_folder
from rlsolver.methods.eco_s2v.src.envs.eeco_util import (SetGraphGenerator,
                                                    RewardSignal, ExtraAction,
                                                    OptimisationTarget, SpinBasis,
                                                    DEFAULT_OBSERVABLES, Observable)
from rlsolver.methods.eco_s2v.src.networks.mpnn import EECO_MPNN as MPNN
from rlsolver.methods.util_read_data import read_nxgraphs
from rlsolver.methods.util_result import write_graph_result
from rlsolver.methods.eco_s2v.config.config import *

def run(save_loc="BA_40spin/eco",
        graph_save_loc="../../data/syn_BA",
        network_save_path=None,
        batched=True,
        max_batch_size=None,
        max_parallel_jobs=4,
        prefixes=INFERENCE_PREFIXES,
        if_greedy=False):
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    data_folder = os.path.join(save_loc)
    print("save location :", data_folder)
    print("network params :", network_save_path)

    ####################################################
    # NETWORK SETUP
    ####################################################

    network_fn = MPNN
    network_args = {
        'n_layers': 3,
        'n_features': 64,
        'n_hid_readout': [],
        'tied_weights': False
    }

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

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
                    'if_greedy':if_greedy}
    if ALG == Alg.s2v:
        env_args = {'observables': [Observable.SPIN_STATE],
                    'reward_signal': RewardSignal.DENSE,
                    'extra_action': ExtraAction.NONE,
                    'optimisation_target': OptimisationTarget.CUT,
                    'spin_basis': SpinBasis.BINARY,
                    'norm_rewards': True,
                    'memory_length': None,
                    'horizon_length': None,
                    'stag_punishment': None,
                    'basin_reward': None,
                    'reversible_spins': False,
                    'if_greedy':if_greedy}

    step_factor = 2
    files = os.listdir(graph_save_loc)

    for prefix in prefixes:
        graphs = []
        file_list = []
        for file in files:
            if prefix in file:
                file = os.path.join(graph_save_loc, file).replace("\\", "/")
                file_list.append(file)
                g = load_graph_from_txt(file)
                g_array = nx.to_numpy_array(g)
                g_tensor = torch.tensor(g_array, dtype=torch.float, device=INFERENCE_DEVICE)
                graphs.append(g_tensor)
        if len(graphs)==0:
            continue
        graphs_test = torch.stack(graphs,dim=0)
        start_time = time.time()
        n_tests = graphs_test.shape[0]
        for start_idx in range(0, n_tests, NUM_INFERENCE_GRAPHS):
            end_idx = min(start_idx + NUM_INFERENCE_GRAPHS, n_tests)
            batch_graphs = graphs_test[start_idx:end_idx]
            file_list_ = file_list[start_idx:end_idx]
            test_graph_generator = SetGraphGenerator(batch_graphs)
            n_inference_graphs = batch_graphs.shape[0]
            ####################################################
            # SETUP NETWORK TO TEST
            ####################################################

            test_env = ising_env.make("SpinSystem",
                                      test_graph_generator,
                                      graphs_test[0].shape[1] * step_factor,  # step_factor is 1 here
                                      **env_args,device = INFERENCE_DEVICE,
                                         n_sims = NUM_INFERENCE_SIMS,
                                         n_graphs = n_inference_graphs)

            network = network_fn(n_obs_in=test_env.observation_space.shape[-1],
                                 **network_args).to(INFERENCE_DEVICE)

            network.load_state_dict(torch.load(network_save_path, map_location=INFERENCE_DEVICE))
            for param in network.parameters():
                param.requires_grad = False
            network.eval()

            # print("Successfully created agent with pre-trained MPNN.\nMPNN architecture\n\n{}".format(repr(network)))
            obj,result = eeco_test_network(network,test_env)
            run_duration = time.time() - start_time
            for i,graph_name in enumerate(file_list_):
                write_graph_result(obj[i], run_duration, result.shape[1], ALG.value, ((result[i]+1)/2).to(torch.int), graph_name, plus1=True)

if __name__ == "__main__":
    prefixes = INFERENCE_PREFIXES
    run(max_parallel_jobs=3, prefixes=prefixes)
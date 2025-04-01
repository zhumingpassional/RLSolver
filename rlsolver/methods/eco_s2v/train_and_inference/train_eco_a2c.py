import matplotlib.pyplot as plt

import rlsolver.methods.eco_s2v.src.envs.core as ising_env
from rlsolver.methods.eco_s2v.util import (cal_txt_name)
from rlsolver.methods.eco_s2v.src.agents.dqn.eeco_a2c import A2C as DQN
from rlsolver.methods.eco_s2v.src.agents.dqn.utils import TestMetric
from rlsolver.methods.eco_s2v.src.envs.eeco_util import (RandomBarabasiAlbertGraphGenerator,
                                                         RandomErdosRenyiGraphGenerator,
                                                         EdgeType, RewardSignal, ExtraAction,
                                                         OptimisationTarget, SpinBasis, ValidationGraphGenerator,
                                                         DEFAULT_OBSERVABLES)
from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN_A2C
from rlsolver.methods.eco_s2v.config import *

try:
    import seaborn as sns

    plt.style.use('seaborn')
except ImportError:
    pass

import time


def run(save_loc, graph_save_loc):
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    gamma = 0.95
    step_fact = 2

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
                }

    ####################################################
    # SET UP TRAINING AND TEST GRAPHS
    ####################################################
    start = time.time()
    n_spins_train = NUM_TRAIN_NODES

    if GRAPH_TYPE == GraphType.ER:
        train_graph_generator = RandomErdosRenyiGraphGenerator(n_spins=n_spins_train, p_connection=0.15,
                                                               edge_type=EdgeType.DISCRETE, n_sims=NUM_TRAIN_SIMS)
    if GRAPH_TYPE == GraphType.BA:
        train_graph_generator = RandomBarabasiAlbertGraphGenerator(n_spins=n_spins_train, m_insertion_edges=4,
                                                                   edge_type=EdgeType.DISCRETE, n_sims=NUM_TRAIN_SIMS,device=TRAIN_DEVICE)

    validation_graph_generator = ValidationGraphGenerator(n_spins=NUM_VALIDATION_NODES, m_insertion_edges=4,
                                                          edge_type=EdgeType.DISCRETE,
                                                          n_sims=NUM_VALIDATION_SIMS, seed=VALIDATION_SEED)
    # validation_graph_generator = RandomBarabasiAlbertGraphGenerator(n_spins=n_spins_train, m_insertion_edges=4,
    #                                                       edge_type=EdgeType.DISCRETE,
    #                                                       n_sims=NUM_VALIDATION_SIMS, 
    #                                                       seed=VALIDATION_SEED,device=TRAIN_DEVICE)    

    ####
    # Pre-generated test graphs
    ####
    graphs_validation = validation_graph_generator.get()
    n_validations = graphs_validation.shape[0]
    # validation_graph_generator = SetGraphGenerator(graphs_validation)
    ####################################################
    # SET UP TRAINING AND TEST ENVIRONMENTS
    ####################################################

    train_envs = ising_env.make("SpinSystem",
                                 train_graph_generator,
                                 int(n_spins_train * step_fact),
                                 **env_args, device=TRAIN_DEVICE,
                                 n_sims=NUM_TRAIN_SIMS)

    n_spins_test = validation_graph_generator.get().shape[1]
    test_envs = ising_env.make("SpinSystem",
                                validation_graph_generator,
                                int(n_spins_test * step_fact),
                                **env_args, device=TRAIN_DEVICE,
                                n_sims=n_validations)

    pre_fix = save_loc + "/" + ALG.value + "_" + GRAPH_TYPE.value + "_" + str(NUM_TRAIN_NODES) + "_" + str(
        NUM_TRAIN_SIMS) + "_"
    pre_fix = cal_txt_name(pre_fix)
    network_save_path = pre_fix + "/network.pth"
    test_save_path = pre_fix + "/test_scores.pkl"
    loss_save_path = pre_fix + "/losses.pkl"
    logger_save_path = pre_fix + f"/logger.json"
    sampling_speed_save_path = pre_fix + "/sampling_speed.json"
    print('pre_fix:', pre_fix.split("/")[-1])
    print(os.path.abspath(sampling_speed_save_path))
    

    ####################################################
    # SET UP AGENT
    ####################################################

    nb_steps = 30000

    network_fn = lambda: MPNN_A2C(n_obs_in=train_envs.observation_space.shape[1],
                              n_layers=3,
                              n_features=64,
                              n_hid_readout=[],
                              tied_weights=False)

    args = {
        'env': train_envs,
        'network': network_fn,
        'device': TRAIN_DEVICE,
    }
    # if TEST_SAMPLING_SPEED:
    #     nb_steps = int(3000)
    #     args['test_frequency'] = args['update_target_frequency'] = args['update_frequency'] = args[
    #         'save_network_frequency'] = 1e6
    #     args['replay_start_size'] = args['initial_exploration_rate'] =0
    #     args['replay_buffer_size'] = NUM_TRAIN_SIMS
    #     args['update_exploration'] = False
    agent = DQN(**args)
    #############
    # TRAIN AGENT
    #############
    sampling_start_time = time.time()
    agent.learn(total_timesteps=nb_steps)


if __name__ == "__main__":
    run()

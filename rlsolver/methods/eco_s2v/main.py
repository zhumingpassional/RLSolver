import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))
from rlsolver.methods.eco_s2v.config.config import *

save_loc = RESULT_DIR

train_network = True
inference_network = False

if train_network:
    if ALG == Alg.eco:
        from rlsolver.methods.eco_s2v.train_and_inference.train_eco import run
    elif ALG == Alg.s2v:
        from rlsolver.methods.eco_s2v.train_and_inference.train_s2v import run
    elif ALG == Alg.eco_torch:
        from rlsolver.methods.eco_s2v.train_and_inference.train_eco_torch import run
    elif ALG == Alg.eeco:
        from rlsolver.methods.eco_s2v.train_and_inference.train_eeco import run
    else:
        raise ValueError('Algorithm not recognized')
    run(save_loc=RESULT_DIR, graph_save_loc=DATA_DIR)


if inference_network:
    if ALG == Alg.eeco:
        from rlsolver.methods.eco_s2v.train_and_inference.inference_eeco import run
        run(graph_folder=DATA_DIR,
        network_folder=NETWORK_FOLDER,
        if_greedy=False,
        n_sims=NUM_INFERENCE_SIMS,
        mini_sims=MINI_INFERENCE_SIMS)
    else:
        from rlsolver.methods.eco_s2v.train_and_inference.inference import run
        run(save_loc=RESULT_DIR, graph_save_loc=DATA_DIR, network_save_path=NETWORK_SAVE_PATH,
            batched=True, max_batch_size=None, max_parallel_jobs=1, prefixes=INFERENCE_PREFIXES)

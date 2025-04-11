import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../')
sys.path.append(os.path.dirname(rlsolver_path))
from rlsolver.methods.jumanji.config import *


train_inference_network = 0  # 0: train, 1: inference
assert train_inference_network in [0, 1]

if train_inference_network == 0:
    from rlsolver.methods.jumanji.train_and_inference.train import run
    run(save_loc=RESULT_DIR, graph_save_loc=DATA_DIR)

if train_inference_network == 1:
    from rlsolver.methods.jumanji.train_and_inference.inference import run
    run(graph_folder=DATA_DIR,n_sims=NUM_INFERENCE_SIMS,mini_sims=MINI_INFERENCE_SIMS)
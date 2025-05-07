from enum import Enum
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../')
from rlsolver.methods.config import GraphType
from rlsolver.methods.util import calc_device
class Alg(Enum):
    eco = 'eco'
    s2v = 's2v'
    eco_torch = 'eco_torch'
    eeco = 'eeco'
    jumanji = 'jumanji'
    rl4co = 'rl4co'

TRAIN_INFERENCE = 0  # 0: train, 1: inference
assert TRAIN_INFERENCE in [0, 1]

ALG = Alg.eeco # Alg
GRAPH_TYPE = GraphType.BA

#训练的参数
TRAIN_GPU_ID = 0
SAMPLE_GPU_ID_IN_ECOS2V = -1 if ALG in [Alg.eco, Alg.s2v] else None
USE_TWO_DEVICES_IN_ECOS2V = True if ALG in [Alg.eco, Alg.s2v] else False
BUFFER_GPU_ID = TRAIN_GPU_ID
NUM_TRAIN_NODES = 200
NUM_TRAIN_SIMS = 2 ** 8
NUM_VALIDATION_NODES = 200
VALIDATION_SEED = 10
NUM_VALIDATION_SIMS = 2 ** 2
TEST_SAMPLING_SPEED = False

#推理的参数
INFERENCE_GPU_ID = 0
NUM_GENERATED_INSTANCES_IN_SELECT_BEST = 3
NUM_INFERENCE_SIMS = 50
USE_LOCAL_SEARCH = True if ALG == Alg.eeco else False
LOCAL_SEARCH_FREQUENCY = 10
MINI_INFERENCE_SIMS = 50  # 如果NUM_INFERENCE_SIMS太大导致GPU内存爆掉，分拆成MINI_INFERENCE_SIMS个环境，跑多次凑够NUM_INFERENCE_SIMS
NUM_INFERENCE_NODES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 2000, 3000, 4000, 5000,10000]
USE_TENSOR_CORE_IN_INFERENCE = True if ALG == Alg.eeco else False
INFERENCE_PREFIXES = [GRAPH_TYPE.value + "_" + str(i) + "_" for i in NUM_INFERENCE_NODES]
NUM_TRAINED_NODES_IN_INFERENCE = 20
# PREFIXES = ["BA_100_", "BA_200_", "BA_300_", "BA_400_", "BA_500_", "BA_600_", "BA_700_", "BA_800_", "BA_900_",
#             "BA_1000_", "BA_1100_", "BA_1200_", "BA_2000_", "BA_3000_", "BA_4000_",
#             "BA_5000_"]  # Replace with your desired prefixes
NEURAL_NETWORK_SAVE_PATH = rlsolver_path + "/methods/eco_s2v/pretrained_agent/" + ALG.value + "_" + GRAPH_TYPE.value + "_" + str(NUM_TRAINED_NODES_IN_INFERENCE) + "spin_best.pth"
DATA_DIR = rlsolver_path + "/data/syn_" + GRAPH_TYPE.value
NEURAL_NETWORK_DIR = rlsolver_path + "/methods/eco_s2v/pretrained_agent/tmp"
NEURAL_NETWORK_FOLDER = rlsolver_path + "/methods/eco_s2v/pretrained_agent/tmp/" + ""
NEURAL_NETWORK_PREFIX = ALG.value + "_" + GRAPH_TYPE.value + "_" + str(NUM_TRAIN_NODES) + "spin"

UPDATE_FREQUENCY = 32
TRAIN_DEVICE = calc_device(TRAIN_GPU_ID)
SAMPLES_DEVICE_IN_ECOS2V = calc_device(SAMPLE_GPU_ID_IN_ECOS2V)
INFERENCE_DEVICE = calc_device(INFERENCE_GPU_ID)
BUFFER_DEVICE = calc_device(BUFFER_GPU_ID)

if GRAPH_TYPE == GraphType.BA:
    if NUM_TRAIN_NODES == 20:
        NB_STEPS = 2500000
        REPLAY_START_SIZE = 500
        REPLAY_BUFFER_SIZE = 5000
        UPDATE_TARGET_FREQUENCY = 1000
        FINAL_EXPLORATION_STEP = 150000
        SAVE_NETWORK_FREQUENCY = 50 # seconds
        TEST_FREQUENCY = 10000
    elif NUM_TRAIN_NODES == 40:
        NB_STEPS = 2500000
        REPLAY_START_SIZE = 500
        REPLAY_BUFFER_SIZE = 5000
        UPDATE_TARGET_FREQUENCY = 1000
        FINAL_EXPLORATION_STEP = 150000
        SAVE_NETWORK_FREQUENCY = 50 # seconds
        TEST_FREQUENCY = 10000
    elif NUM_TRAIN_NODES == 60 or NUM_TRAIN_NODES == 80:
        NB_STEPS = 5000000
        REPLAY_START_SIZE = 500
        REPLAY_BUFFER_SIZE = 5000
        UPDATE_TARGET_FREQUENCY = 1000
        FINAL_EXPLORATION_STEP = 300000
        SAVE_NETWORK_FREQUENCY = 50 # seconds
        TEST_FREQUENCY = 20000
    elif NUM_TRAIN_NODES == 100:
        NB_STEPS = 8000000
        REPLAY_START_SIZE = 1500
        REPLAY_BUFFER_SIZE = 10000
        UPDATE_TARGET_FREQUENCY = 2500
        FINAL_EXPLORATION_STEP = 800000
        SAVE_NETWORK_FREQUENCY = 500 # seconds
        TEST_FREQUENCY = 50000
    elif NUM_TRAIN_NODES >= 200:
        NB_STEPS = 10000000
        REPLAY_START_SIZE = NUM_TRAIN_NODES*2*NUM_TRAIN_SIMS
        REPLAY_BUFFER_SIZE = 5*NUM_TRAIN_NODES*2*NUM_TRAIN_SIMS
        UPDATE_TARGET_FREQUENCY = 4000
        FINAL_EXPLORATION_STEP = 800000
        SAVE_NETWORK_FREQUENCY = 500 # seconds
        TEST_FREQUENCY = 4000000
    else:
        raise ValueError("parameters are not set")
elif GRAPH_TYPE == GraphType.ER:
    if NUM_TRAIN_NODES == 20:
        NB_STEPS = 2500000
        REPLAY_START_SIZE = 500
        REPLAY_BUFFER_SIZE = 5000
        UPDATE_TARGET_FREQUENCY = 1000
        FINAL_EXPLORATION_STEP = 150000
        SAVE_NETWORK_FREQUENCY = 50 # seconds
        TEST_FREQUENCY = 10000
    elif NUM_TRAIN_NODES == 40:
        NB_STEPS = 2500000
        REPLAY_START_SIZE = 500
        REPLAY_BUFFER_SIZE = 5000
        UPDATE_TARGET_FREQUENCY = 1000
        FINAL_EXPLORATION_STEP = 150000
        SAVE_NETWORK_FREQUENCY = 50 # seconds
        TEST_FREQUENCY = 10000
    elif NUM_TRAIN_NODES == 60:
        NB_STEPS = 5000000
        REPLAY_START_SIZE = 500
        REPLAY_BUFFER_SIZE = 5000
        UPDATE_TARGET_FREQUENCY = 1000
        FINAL_EXPLORATION_STEP = 300000
        SAVE_NETWORK_FREQUENCY = 50 # seconds
        TEST_FREQUENCY = 20000
    elif NUM_TRAIN_NODES == 100:
        NB_STEPS = 8000000
        REPLAY_START_SIZE = 1500
        REPLAY_BUFFER_SIZE = 10000
        UPDATE_TARGET_FREQUENCY = 2500
        FINAL_EXPLORATION_STEP = 800000
        SAVE_NETWORK_FREQUENCY = 50 # seconds
        TEST_FREQUENCY = 50000
    elif NUM_TRAIN_NODES >= 200:
        NB_STEPS = 10000000
        REPLAY_START_SIZE = 3000
        REPLAY_BUFFER_SIZE = 70000
        UPDATE_TARGET_FREQUENCY = 4000
        FINAL_EXPLORATION_STEP = 800000
        SAVE_NETWORK_FREQUENCY = 500 # seconds
        TEST_FREQUENCY = 50000
    else:
        raise ValueError("parameters are not set")

#jumanji
HERIZON_LENGTH = int(NUM_TRAIN_NODES/2)
JUMANJI_TEST_FREQUENCY = 10 #每次test的时间间隔

#rl4co
RL4CO_GRAPH_DIR = rlsolver_path + "/data/syn_BA/BA_100_ID0.txt"
RL4CO_CHECKOUT_DIR = rlsolver_path + "/methods/eco_s2v/pretrained_agent/tmp/rl4co_BA_20spin/rl4co_BA_20spin_step=000250.ckpt"

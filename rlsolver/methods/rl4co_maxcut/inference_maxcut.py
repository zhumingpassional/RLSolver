import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver/methods/rl4co_maxcut')
sys.path.append(os.path.dirname(rlsolver_path))

from rl4co_maxcut.envs.graph import MaxCutEnv, MaxCutGenerator
from rl4co_maxcut.models.nn.graph.gcn import GCNEncoder
from rl4co_maxcut.models import AttentionModelPolicy, AttentionModel
from rl4co_maxcut.utils import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
import torch

device = "cuda:4"
checkpoint_path = "rlsolver/methods/rl4co_maxcut/checkpoint/maxcut_step_step=000500.ckpt"
new_model_checkpoint = AttentionModel.load_from_checkpoint(checkpoint_path, strict=False)
policy_new = new_model_checkpoint.policy.to(device)
generator = MaxCutGenerator(n_spins=20, loc_distribution="uniform")
env = MaxCutEnv(generator).to(device)
td_init = env.reset(batch_size=[64]).to(device)
out = policy_new(td_init.clone(), env, phase="test", decode_type="greedy")
print(out['reward'])
print(torch.mean(out['reward']))
print(torch.max(out['reward']))
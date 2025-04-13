import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../')
sys.path.append(os.path.dirname(rlsolver_path))

from rlsolver.methods.rl4co_maxcut.envs.graph import MaxCutEnv
from rlsolver.methods.rl4co_maxcut.envs.graph.maxcut.inference_generator import MaxCutGenerator
from rlsolver.methods.rl4co_maxcut.models import S2VModel
import torch
torch.set_grad_enabled(False)
file_name = rlsolver_path+"rlsolver/data/syn_BA/BA_100_ID0.txt"
device = "cuda:7"
checkpoint_path = rlsolver_path + "rlsolver/methods/rl4co_maxcut/checkpoints/maxcut_step_step=000250.ckpt"
new_model_checkpoint = S2VModel.load_from_checkpoint(checkpoint_path, strict=False,map_location=device)

policy_new = new_model_checkpoint.policy.to(device)
generator = MaxCutGenerator(file=file_name,device=device)
env = MaxCutEnv(generator).to(device)
td_init = env.reset(batch_size=[64]).to(device)

out = policy_new(td_init.clone(), env, phase="test", decode_type="greedy")
out['reward'] = out['reward']*generator.n_spins
print(out['reward'])
print(torch.mean(out['reward']))
print(torch.max(out['reward']))
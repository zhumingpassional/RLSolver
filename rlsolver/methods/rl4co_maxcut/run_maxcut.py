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


# Instantiate generator and environment
generator = MaxCutGenerator(n_spins=20)
env = MaxCutEnv(generator)
gcn_encoder = GCNEncoder(
    env_name='maxcut', 
    embed_dim=64,
    num_layers=3,
)

# Create policy and RL model
policy = AttentionModelPolicy(env_name=env.name, embed_dim=64,num_encoder_layers=6)

model = AttentionModel(env,policy,batch_size=64,train_data_size=90000, optimizer_kwargs={"lr": 1e-4},policy_kwargs={
        'encoder': gcn_encoder
    })


checkpoint_callback = ModelCheckpoint(
    dirpath="rlsolver/methods/rl4co_maxcut/checkpoint",  # 保存路径
    filename="maxcut_step_{step:06d}",  # 按步数保存
    every_n_train_steps=500,  # 每 1000 步保存一次
    save_top_k=-1,  # 保存所有模型
    save_last=False,  # 保存最后一个模型
    mode="max",  # 最大化 reward
)
rich_model_summary = RichModelSummary(max_depth=3)  # model summary callback
callbacks = [checkpoint_callback, rich_model_summary]

trainer = RL4COTrainer(max_epochs=1, accelerator="gpu",precision="16-mixed",callbacks=callbacks,devices=[4])
trainer.fit(model)
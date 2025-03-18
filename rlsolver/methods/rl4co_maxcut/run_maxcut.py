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

device = "cuda:3"
# Instantiate generator and environment
generator = MaxCutGenerator()
env = MaxCutEnv(generator)

gcn_encoder = GCNEncoder(
    env_name='maxcut', 
    embed_dim=256,
    num_layers=3,
)
# Create policy and RL model
policy = AttentionModelPolicy(env_name=env.name, embed_dim=256,num_encoder_layers=6)
model = AttentionModel(env,policy,batch_size=64,train_data_size=1000, baseline="rollout", optimizer_kwargs={"lr": 1e-4},policy_kwargs={
        'encoder': gcn_encoder
    })
#20000000
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",  # save to checkpoints/
    filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
    save_top_k=1,  # save only the best model
    save_last=True,  # save the last model
    monitor="val/reward",  # monitor validation reward
    mode="max",
)  # maximize validation reward
rich_model_summary = RichModelSummary(max_depth=3)  # model summary callback
callbacks = [checkpoint_callback, rich_model_summary]
# Instantiate Trainer and fit
trainer = RL4COTrainer(max_epochs=1, accelerator="gpu", reload_dataloaders_every_n_epochs=2,precision="16-mixed",callbacks=callbacks,devices=[0])
trainer.fit(model)
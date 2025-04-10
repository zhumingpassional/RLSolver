import os
import numpy as np
import torch as th
from torch import nn
from torch.nn.utils import clip_grad_norm_
from typing import Union, Optional
from rlsolver.methods.jumanji.config import *


TEN = th.Tensor

'''agent'''


class AgentBase:
    """
    The basic agent of ElegantRL

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):

        self.gamma = args.gamma  # discount factor of future rewards
        self.max_step = args.max_step  # limits the maximum number of steps an agent can take in a trajectory.
        self.num_envs = NUM_TRAIN_SIMS  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state
        self.buffer_init_size = args.buffer_init_size  # train after samples over buffer_init_size for off-policy

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''network'''
        self.act = None
        self.cri = None
        self.act_target = self.act
        self.cri_target = self.cri

        '''optimizer'''
        self.act_optimizer: Optional[th.optim] = None
        self.cri_optimizer: Optional[th.optim] = None

        self.criterion = getattr(args, 'criterion', th.nn.MSELoss(reduction="none"))
        self.if_vec_env = self.num_envs > 1  # use vectorized environment (vectorized simulator)
        self.if_use_per = getattr(args, 'if_use_per', None)  # use PER (Prioritized Experience Replay)
        self.lambda_fit_cum_r = getattr(args, 'lambda_fit_cum_r', 0.0)  # critic fits cumulative returns

        """save and load"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

    def explore_env(self, env, horizon_len: int) -> tuple[TEN, TEN, TEN, TEN, TEN]:
        return self._explore_vec_env(env=env, horizon_len=horizon_len)

    def explore_action(self, state: TEN) -> TEN:
        return self.act.get_action(state)

    def _explore_vec_env(self, env, horizon_len: int) -> tuple[TEN, TEN, TEN, TEN, TEN]:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, rewards, undones, unmasks)` for off-policy
            `num_envs > 1`
            `states.shape == (horizon_len, num_envs, state_dim)`
            `actions.shape == (horizon_len, num_envs, action_dim)`
            `rewards.shape == (horizon_len, num_envs)`
            `undones.shape == (horizon_len, num_envs)`
            `unmasks.shape == (horizon_len, num_envs)`
        """
        states = th.zeros((horizon_len, self.num_envs, NUM_TRAIN_NODES+7,NUM_TRAIN_NODES), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.num_envs), dtype=th.long).to(self.device)
        rewards = th.zeros((horizon_len, self.num_envs), dtype=th.float32).to(self.device)
        terminals = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)
        truncates = th.zeros((horizon_len, self.num_envs), dtype=th.bool).to(self.device)

        state = env.get_observation()  # last_state.shape == (num_envs, state_dim)
        for t in range(horizon_len):
            action,logprobs = self.explore_action(state)

            states[t] = state  # state.shape == (num_envs, state_dim)
            actions[t] = action

            state, reward, terminal, truncate, _ = env.step(action)

            rewards[t] = reward
            terminals[t] = terminal
            truncates[t] = truncate

        self.last_state = state
        undones = th.logical_not(terminals)
        unmasks = th.logical_not(truncates)
        return states, actions, logprobs, rewards, undones, unmasks

    def update_net(self, buffer: Union[ReplayBuffer, tuple]) -> tuple[float, ...]:
        objs_critic = []
        objs_actor = []

        obj_critic, obj_actor = self.update_objectives(buffer=buffer, update_t=update_t)
        objs_critic.append(obj_critic)
        objs_actor.append(obj_actor) if isinstance(obj_actor, float) else None
        th.set_grad_enabled(False)

        obj_avg_critic = np.nanmean(objs_critic) if len(objs_critic) else 0.0
        obj_avg_actor = np.nanmean(objs_actor) if len(objs_actor) else 0.0
        return obj_avg_critic, obj_avg_actor

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> tuple[float, float]:
        assert isinstance(update_t, int)
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state,
                 is_weight, is_index) = buffer.sample_for_per(self.batch_size)
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None

            next_action = self.act(next_state)  # deterministic policy
            next_q = self.cri_target(next_state, next_action)

            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state, action) * unmask
        td_error = self.criterion(q_value, q_label) * unmask
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        if_update_act = bool(buffer.cur_size >= self.buffer_init_size)
        if if_update_act:
            action_pg = self.act(state)  # action to policy gradient
            obj_actor = self.cri(state, action_pg).mean()
            self.optimizer_backward(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        else:
            obj_actor = th.tensor(th.nan)
        return obj_critic.item(), obj_actor.item()

    def get_cumulative_rewards(self, rewards: TEN, undones: TEN) -> TEN:
        cum_rewards = th.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_action = self.act_target(last_state)
        next_value = self.cri_target(last_state, next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            cum_rewards[t] = next_value = rewards[t] + masks[t] * next_value
        return cum_rewards

    def optimizer_backward(self, optimizer: th.optim, objective: TEN):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = th.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def optimizer_backward_amp(self, optimizer: th.optim, objective: TEN):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = th.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = th.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from th.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net: th.nn.Module, current_net: th.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"

            if getattr(self, attr_name) is None:
                continue

            if if_save:
                th.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name, th.load(file_path, map_location=self.device))


def get_optim_param(optimizer: th.optim) -> list:  # backup
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, th.Tensor)])
    return params_list


'''network'''


class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = None  # build_mlp(net_dims=[state_dim, *net_dims, action_dim])

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = th.distributions.normal.Normal

    def forward(self, state: TEN) -> TEN:
        action = self.net(state)
        return action.tanh()


class CriticBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(net_dims=[state_dim + action_dim, *net_dims, 1])

    def forward(self, state: TEN, action: TEN) -> TEN:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=-1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        values = self.net(th.cat((state, action), dim=1))
        return values  # Q values



def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


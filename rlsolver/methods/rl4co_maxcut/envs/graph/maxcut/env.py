from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rlsolver.methods.rl4co_maxcut.envs.common.base import RL4COEnvBase
from rlsolver.methods.rl4co_maxcut.utils.ops import gather_by_index
from rlsolver.methods.rl4co_maxcut.utils.pylogger import get_pylogger

from .generator import MaxCutGenerator

log = get_pylogger(__name__)


class MaxCutEnv(RL4COEnvBase):

    name = "maxcut"

    def __init__(
        self,
        generator: MaxCutGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MaxCutGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # action: [batch_size, 1]; the location to be chosen in each instance
        selected = td["action"]
        batch_size = selected.shape[0]

        # Update location selection status
        state = td["state"].clone()  # (batch_size, n_locations)

        state[torch.arange(batch_size).to(td.device), selected] = torch.logical_not(td["state"][torch.arange(batch_size).to(td.device), selected])
        td["state"] = state
        # We are done if we choose enough locations
        done = td["i"] >= (td["to_choose"] - 1)
        action_mask = torch.ones_like(td["state"], dtype=torch.bool)  # 允许所有动作

        # The reward is calculated outside via get_reward for efficiency, so we set it to zero here
        reward = torch.zeros_like(done)
        # sim_indices = torch.arange(batch_size,device=td.device)
        state_ = state*2-1
        greedy_change = (torch.matmul(td["adj"], state_.to(torch.float).unsqueeze(-1)).squeeze(-1) * state_.to(torch.float))
        # Update distances

        td.update(
            {   "state":state,
                # states changed by actions
                "i": td["i"] + 1,  # the number of sets we have chosen
                "reward": reward,
                "done": done,
                "adj": td["adj"],
                "action_mask" : action_mask,
                "greedy_change": greedy_change,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:

        self.to(td.device)
        # state_ = torch.randint(0, 2, (*batch_size, self.generator.n_spins), dtype=torch.bool, device=td.device)
        state_ = torch.ones((*batch_size,self.generator.n_spins), dtype=torch.bool,device=td.device)
        greedy_change = (torch.matmul(td["adj"], state_.to(torch.float).unsqueeze(-1)).squeeze(-1) * state_.to(torch.float))

        return TensorDict(
            {
                # given information
                "adj": td["adj"],  # (batch_size, n_points, dim_loc)
                "to_choose": td["to_choose"],  # 每个环境的交互次数
                "state": state_,
                "i": torch.zeros(
                    *batch_size, dtype=torch.int64, device=td.device
                ),
                "action_mask" : torch.ones_like(td["state"], dtype=torch.bool),
                "greedy_change": greedy_change,
            },
            batch_size=batch_size,
        )        

    def _make_spec(self, generator: MaxCutGenerator):
        self.action_spec = Bounded(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.n_spins,
        )

    def _get_reward(self, td: TensorDict,actions) -> torch.Tensor:
        # breakpoint()
        state = td['state'].to(torch.float)*2-1
        obj = 0.1*((1 / 4) * (torch.matmul(td['adj'],state.unsqueeze(-1)).squeeze(-1) * -state).sum(dim=-1) + (1 / 4)
                        * torch.sum(td['adj'],dim=(-1,-2)))
        
        # print(state.shape,td['adj'].shape)
        return obj

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        # TODO: check solution validity
        pass

    @staticmethod
    def local_search(td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        # TODO: local search
        pass

    @staticmethod
    def get_num_starts(td):
        return td["action_mask"].shape[-1]

    @staticmethod
    def select_start_nodes(td, num_starts):
        num_loc = td["action_mask"].shape[-1]
        return (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc
        )

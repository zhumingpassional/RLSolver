from rlsolver.methods.rl4co_maxcut.envs.common.base import RL4COEnvBase
from rlsolver.methods.rl4co_maxcut.models.rl import REINFORCE
from rlsolver.methods.rl4co_maxcut.models.rl.reinforce.baselines import REINFORCEBaseline
from rlsolver.methods.rl4co_maxcut.models.zoo.am.policy import AttentionModelPolicy


class AttentionModel(REINFORCE):
    """Attention Model based on REINFORCE: https://arxiv.org/abs/1803.08475.
    Check :class:`REINFORCE` and :class:`rl4co.models.RL4COLitModule` for more details such as additional parameters  including batch size.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to rollout (1 epoch of exponential, then greedy rollout baseline)
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: AttentionModelPolicy = None,
        baseline: REINFORCEBaseline | str = "rollout",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = AttentionModelPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)

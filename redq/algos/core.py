from typing import Dict, Optional, Tuple, Union, Any

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

from redq.algos.sumtree import SumTree

# following SAC authors' and OpenAI implementation
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6
# these numbers are from the MBPO paper
mbpo_target_entropy_dict = {'Hopper-v2': -1, 'HalfCheetah-v2': -3, 'Walker2d-v2': -3, 'Ant-v2': -4, 'Humanoid-v2': -2}
mbpo_epoches = {'Hopper-v2': 125, 'Walker2d-v2': 300, 'Ant-v2': 300, 'HalfCheetah-v2': 400, 'Humanoid-v2': 300}


class ReplayBuffer:
    """
    A FIFO experience replay buffer with optional prioritized sampling.

    replay_mode:
      - "uniform"       : legacy behavior (default)
      - "static_pitod"  : alias of "uniform"; kept so experiment labeling is clean
      - "per"           : Prioritized Experience Replay (Schaul et al. 2015).
                          Training loop writes |TD|+eps back via update_priorities.
      - "dynamic_pitod" : SumTree priorities managed externally by
                          DynamicPIToDController (group-level rescoring).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        size: int,
        experience_group_size: int = 5000,
        mask_dim: int = 20,
        replay_mode: str = "uniform",
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_anneal_steps: int = 1_000_000,
        per_epsilon: float = 1e-6,
        sumtree: Optional[SumTree] = None,
    ) -> None:
        """
        :param obs_dim: size of observation
        :param act_dim: size of the action
        :param size: size of the buffer
        :param experience_group_size: size of experience group
        :param mask_dim: size of mask for turn-over dropout
        :param replay_mode: sampling strategy (see class docstring)
        :param per_alpha: PER priority exponent
        :param per_beta_start: starting IS-weight exponent
        :param per_beta_end: final IS-weight exponent
        :param per_beta_anneal_steps: linear anneal horizon (in env steps)
        :param per_epsilon: small constant added to PER priorities
        :param sumtree: shared SumTree used by PER and Dynamic PIToD; owned externally
        """

        # init buffers as numpy arrays
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.masks_buf = np.zeros([size, mask_dim], dtype=np.int32) # init buffers mask.
        self.ptr, self.size, self.max_size = 0, 0, size

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.experience_group_size = experience_group_size
        self.mask_dim = mask_dim

        # --- prioritized-sampling state ---
        self.replay_mode = replay_mode
        self.sumtree = sumtree
        self.per_alpha = float(per_alpha)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_end = float(per_beta_end)
        self.per_beta_anneal_steps = int(per_beta_anneal_steps)
        self.per_epsilon = float(per_epsilon)
        self._sample_rng = np.random.RandomState()
        self._steps_seen = 0  # env-steps stored; used only for beta annealing

        if replay_mode in ("per", "dynamic_pitod") and sumtree is None:
            raise ValueError(f"replay_mode={replay_mode!r} requires a SumTree to be passed in")

    def _current_beta(self) -> float:
        if self.per_beta_anneal_steps <= 0:
            return self.per_beta_end
        frac = min(max(self._steps_seen / self.per_beta_anneal_steps, 0.0), 1.0)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    def store(self, obs: np.ndarray, act: np.ndarray, rew: np.float64, next_obs: np.ndarray, done: bool) -> None:
        """
        data will get stored in the pointer's location
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        # create and store mask associated for the data
        if (self.ptr % self.experience_group_size) == 0:
            # dropout rate of ToD = 0.5
            self.ids_zero_elem = np.random.permutation(self.mask_dim)[:int(self.mask_dim / 2)]
        mask = np.ones([self.mask_dim], dtype=np.int32)
        mask[self.ids_zero_elem] = 0.0
        self.masks_buf[self.ptr] = mask

        # priority bookkeeping (new transitions get optimistic priority so they get sampled once)
        if self.sumtree is not None:
            self.sumtree.update(self.ptr, self.sumtree.max_priority)

        # move the pointer to store in next location in buffer
        self.ptr = (self.ptr + 1) % self.max_size
        # keep track of the current buffer size
        self.size = min(self.size + 1, self.max_size)
        self._steps_seen += 1

    def sample_batch(self, batch_size: int = 32, idxs: Union[np.ndarray, None] = None) -> Dict:
        """
        :param batch_size: size of minibatch
        :param idxs: specify indexes if you want specific data points
        :return: mini-batch data as a dictionary
        """
        is_weights: Optional[np.ndarray] = None
        if idxs is not None:
            # explicit indices path (used by bias_utils & dynamic rescoring) — always uniform weighting
            pass
        elif self.replay_mode in ("per", "dynamic_pitod") and self.sumtree is not None and self.sumtree.total() > 0:
            idxs, priorities = self.sumtree.sample(batch_size, self._sample_rng)
            total = self.sumtree.total()
            probs = np.maximum(priorities, 1e-12) / total
            beta = self._current_beta()
            is_weights = (self.size * probs) ** (-beta)
            is_weights = is_weights / is_weights.max()
            is_weights = is_weights.astype(np.float32)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)

        if is_weights is None:
            is_weights = np.ones(len(idxs), dtype=np.float32)

        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    masks=self.masks_buf[idxs],
                    idxs=idxs,
                    is_weights=is_weights)

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray) -> None:
        """Write priorities back to the SumTree (PER writeback path)."""
        if self.sumtree is None:
            return
        priorities = (np.abs(priorities) + self.per_epsilon) ** self.per_alpha
        for data_idx, p in zip(np.asarray(idxs).reshape(-1), priorities.reshape(-1)):
            self.sumtree.update(int(data_idx), float(p))


class Mlp(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_sizes: Tuple[int, ...],
            hidden_activation: Any = F.relu,
            # following are turn-over dropout related.
            target_drop_rate: float = 0.0,
            layer_norm: bool = False,
            ensemble_size: int = 20,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.hidden_sizes = hidden_sizes
        self.target_drop_rate = target_drop_rate
        self.apply_layer_norm = layer_norm
        self.ensemble_size = ensemble_size

        hidden_input_size = input_size
        # weight
        self.hidden_weights = []
        for i in range(0, len(hidden_sizes)):
            param = nn.Parameter(torch.empty(self.ensemble_size, hidden_input_size, hidden_sizes[i]))
            torch.nn.init.xavier_uniform_(param, gain=1)
            self.hidden_weights.append(param)
            hidden_input_size = hidden_sizes[i]
        self.hidden_weights = torch.nn.ParameterList(self.hidden_weights)
        self.output_weight = nn.Parameter(torch.empty(self.ensemble_size, hidden_input_size, output_size))
        torch.nn.init.xavier_uniform_(self.output_weight, gain=1)
        # bias
        self.hidden_biases = []
        for ind in range(0, len(hidden_sizes)):
            param = nn.Parameter(torch.empty(self.ensemble_size, 1, hidden_sizes[ind]))
            torch.nn.init.xavier_uniform_(param, gain=1)
            self.hidden_biases.append(param)
        self.hidden_biases = torch.nn.ParameterList(self.hidden_biases)
        self.output_bias = nn.Parameter(torch.empty(self.ensemble_size, 1, output_size))
        torch.nn.init.constant_(self.output_bias, 0)
        # Layer normalization and dropout
        if self.apply_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_sizes[0])
        if self.target_drop_rate > 0.0:
            self.dropout = nn.Dropout(self.target_drop_rate)

    def forward(self, input: torch.Tensor, masks: Union[torch.Tensor, None] = None, flips: bool = False) \
            -> torch.Tensor:
        hidden = input.unsqueeze(0).expand(self.ensemble_size, -1, -1)  # [n_mlps, batch_size, input_size]

        # forward
        for hidden_weight, hidden_bias in zip(self.hidden_weights, self.hidden_biases):
            hidden = torch.bmm(hidden, hidden_weight) + hidden_bias  # [n_mlps, batch_size, hidden_size]
            hidden = hidden.view(-1, hidden.size(-1))  # [n_mlps * batch_size, hidden_size]
            if self.target_drop_rate > 0.0:
                hidden = self.dropout(hidden)
            if self.apply_layer_norm:
                hidden = self.layer_norm(hidden)
            hidden = self.hidden_activation(hidden)
            hidden = hidden.view(self.ensemble_size, -1, hidden.size(-1))  # [n_mlps, batch_size, hidden_size]
        output = torch.bmm(hidden, self.output_weight) + self.output_bias  # [n_mlps, batch_size, output_size]

        # masking
        if masks is not None:
            masks_t = torch.transpose(masks, 0, 1).unsqueeze(
                -1)  # [batch_size, masks_size] -> [mask_size, batch_size, 1]
        else:
            if not hasattr(self, "_default_mask"):
                self._default_mask = torch.ones(self.ensemble_size, 1, 1).to(input.device)  # [mask_size, 1, 1]
            masks_t = self._default_mask.expand(self.ensemble_size, input.shape[0], 1)  #  [mask_size, batch_size, 1]
        if (masks is not None) and flips:
            masks_t = (1.0 - masks_t)

        # weighted sum
        masked_output = torch.sum(output * masks_t, dim=0) / torch.sum(masks_t, dim=0)

        return masked_output


class TanhGaussianPolicy(Mlp):
    """
    A Gaussian policy network with Tanh to enforce action limits
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_sizes: Tuple[int, ...],
            hidden_activation: Any = F.relu,
            action_limit: float = 1.0,
            layer_norm: bool = False,
            ensemble_size: int = 20,
    ) -> None:
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            layer_norm=layer_norm,
            ensemble_size=ensemble_size
        )
        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        # this is the layer that gives log_std, init this layer with small weight and bias
        self.log_std_weight = nn.Parameter(torch.empty(self.ensemble_size,
                                                       last_hidden_size,
                                                       action_dim))
        torch.nn.init.xavier_uniform_(self.log_std_weight, gain=1)
        self.log_std_bias = nn.Parameter(torch.empty(self.ensemble_size,
                                                     1,
                                                     action_dim))
        torch.nn.init.constant_(self.log_std_bias, 0)

        # action limit: for example, humanoid has an action limit of -0.4 to 0.4
        self.action_limit = action_limit

    def forward(self, obs: torch.Tensor,
                deterministic: bool = False,
                return_log_prob: bool = True,
                masks: Union[torch.Tensor, None] = None,
                flips: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor]:
        """
        :param obs: Observation
        :param deterministic: If True, take deterministic (test) action
        :param return_log_prob: If True, return a sample and its log probability
        """
        hidden = obs.unsqueeze(0).expand(self.ensemble_size, -1, -1)  # [n_mlps, batch_size, input_size]

        # forward
        for hidden_weight, hidden_bias in zip(self.hidden_weights, self.hidden_biases):
            hidden = torch.bmm(hidden, hidden_weight) + hidden_bias  # [n_mlps, batch_size, hidden_size]
            hidden = hidden.view(-1, hidden.size(-1))  # [n_mlps * batch_size, hidden_size]
            if self.target_drop_rate > 0.0:
                hidden = self.dropout(hidden)
            if self.apply_layer_norm:
                hidden = self.layer_norm(hidden)
            hidden = self.hidden_activation(hidden)
            hidden = hidden.view(self.ensemble_size, -1, hidden.size(-1))  # [n_mlps, batch_size, hidden_size]
        output = torch.bmm(hidden, self.output_weight) + self.output_bias  # [n_mlps, batch_size, output_size]
        log_std = torch.bmm(hidden, self.log_std_weight) + self.log_std_bias

        # masking
        if masks is not None:
            masks_t = torch.transpose(masks, 0, 1).unsqueeze(
                -1)  # [batch_size, masks_size] -> [mask_size, batch_size, 1]
        else:
            if not hasattr(self, "_default_mask"):
                self._default_mask = torch.ones(self.ensemble_size, 1, 1).to(obs.device)  # [mask_size, 1, 1]
            masks_t = self._default_mask.expand(self.ensemble_size, obs.shape[0], 1)  # [mask_size, batch_size, 1]
        if (masks is not None) and flips:
            masks_t = (1.0 - masks_t)

        # weighted sum
        masked_output = torch.sum(output * masks_t, dim=0) / torch.sum(masks_t, dim=0)
        masked_log_std = torch.sum(log_std * masks_t, dim=0) / torch.sum(masks_t, dim=0)

        mean = masked_output
        log_std = masked_log_std
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
        )


def soft_update_model1_with_model2(model1: nn.Module, model2: nn.Module, rou: float) -> None:
    """
    used to polyak update a target network
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou * model1_param.data + (1 - rou) * model2_param.data)


def test_agent(agent: Any,
               test_env: gym.Env,
               max_ep_len: int,
               logger: Any,
               n_eval: int = 1) -> np.ndarray:
    """
    This will test the agent's performance by running <n_eval> episodes
    During the runs, the agent should only take deterministic action
    This function assumes the agent has a <get_test_action()> function
    :param agent: agent instance
    :param test_env: the environment used for testing
    :param max_ep_len: max length of an episode
    :param logger: logger to store info in
    :param n_eval: number of episodes to run the agent
    :return: test return for each episode as a numpy array
    """
    ep_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_test_action(o)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        if logger is not None:
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    return ep_return_list

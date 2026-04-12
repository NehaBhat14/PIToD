"""
Dynamic PIToD entry script.

Drop-in replacement for main-TH.py adding four replay modes:
  --replay_mode {uniform, per, static_pitod, dynamic_pitod}

For dynamic_pitod, every K environment steps the DynamicPIToDController
rescores a batch of older experience groups using current network weights,
broadcasts priorities into a SumTree, and optionally prunes low-scoring
groups. For per, the training loop writes |TD error|+eps back after each
update. For uniform / static_pitod, sampling is uniform as in main-TH.py.

See CS6955_Project_Proposal.pdf and DYNAMIC_PITOD_PLANNING.md for the design.
"""

from typing import Dict, Tuple, Union
import os
import sys
import time

import gym
import numpy as np
import torch

from redq.algos.redq_sac import REDQSACAgent
from redq.algos.core import mbpo_epoches, test_agent
from redq.algos.sumtree import SumTree
from redq.algos.group_registry import GroupRegistry
from redq.utils.run_utils import setup_logger_kwargs
from redq.utils.bias_utils import log_evaluation
from redq.utils.dynamic_pitod_utils import DynamicPIToDController, H2Tracker
from redq.utils.logx import EpochLogger

import customenvs
customenvs.register_mbpo_environments()
dm_control_env = ["fish-swim", "hopper-hop", "quadruped-run",
                  "cheetah-run", "humanoid-run", "humanoid-stand",
                  "finger-turn_hard", "hopper-stand"]


def dynamic_pitod(
    env_name: str,
    seed: int = 0,
    epochs: int = -1,
    steps_per_epoch: int = 5000,
    max_ep_len: int = 1000,
    n_evals_per_epoch: int = 10,
    adversarial_reward_epoch: int = -999,
    logger_kwargs: Dict = dict(),
    gpu_id: int = 0,
    # base agent hyperparameters
    hidden_sizes: Tuple[int, ...] = (128, 128),
    replay_size: int = int(1.51e6),
    batch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    polyak: float = 0.995,
    alpha: float = 0.2,
    auto_alpha: bool = True,
    target_entropy: Union[str, float] = 'mbpo',
    start_steps: int = 5000,
    delay_update_steps: Union[str, int] = 'auto',
    utd_ratio: int = 4,
    num_Q: int = 2,
    num_min: int = 2,
    policy_update_delay: int = 20,
    # bias-evaluation knobs
    evaluate_bias: bool = False,
    n_mc_eval: int = 1000,
    n_mc_cutoff: int = 350,
    reseed_each_epoch: bool = True,
    # network structure
    layer_norm: bool = False,
    layer_norm_policy: bool = False,
    experience_group_size: int = 5000,
    mask_dim: int = 20,
    target_drop_rate: float = 0.0,
    reset_interval: int = -1,
    # static-PIToD evaluation (off by default in dynamic mode to save time)
    experience_cleansing: bool = False,
    dump_trajectory_for_demo: bool = False,
    record_training_self_training_losses: bool = False,
    influence_estimation_interval: int = 10,
    n_eval: int = 10,
    # --- Dynamic PIToD / PER knobs ---
    replay_mode: str = 'uniform',
    k_refresh: int = 5000,
    b_refresh: int = 32,
    m_strikes: int = 3,
    epsilon_k: float = 1.0,
    pitod_alpha: float = 0.6,
    n_samples_per_group: int = 64,
    dynamic_warmup_steps: int = 5000,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_end: float = 1.0,
    per_beta_anneal_steps: int = 1_000_000,
    # H2 logging
    h2_log: bool = True,
    h2_tag_step: int = 10_000,
    h2_tag_n_groups: int = 2,
):
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    if epochs < 0:
        mbpo_epoches['AntTruncatedObs-v2'] = 300
        mbpo_epoches['HumanoidTruncatedObs-v2'] = 300
        mbpo_epoches.update(dict(zip(dm_control_env, [300 for _ in dm_control_env])))
        epochs = mbpo_epoches[env_name]
    total_steps = steps_per_epoch * epochs + 1

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    if env_name in dm_control_env:
        import dmc2gym
        domain_name, task_name = env_name.split("-")[0], env_name.split("-")[1]
        env = dmc2gym.make(domain_name, task_name)
        test_env = dmc2gym.make(domain_name, task_name)
        bias_eval_env = dmc2gym.make(domain_name, task_name)
        if target_entropy == "mbpo":
            target_entropy = 'auto'
    else:
        env, test_env, bias_eval_env = gym.make(env_name), gym.make(env_name), gym.make(env_name)

    torch.manual_seed(seed)
    np.random.seed(seed)

    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)

    seed_all(epoch=0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    act_limit = env.action_space.high[0].item()
    start_time = time.time()
    sys.stdout.flush()

    # --- Build SumTree + Registry before agent, then agent holds a reference.
    use_sumtree = replay_mode in ("per", "dynamic_pitod")
    sumtree = SumTree(capacity=replay_size) if use_sumtree else None
    registry = (
        GroupRegistry(buffer_capacity=replay_size, experience_group_size=experience_group_size)
        if replay_mode == "dynamic_pitod" else None
    )

    agent = REDQSACAgent(
        env_name=env_name, obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, device=device,
        hidden_sizes=hidden_sizes, replay_size=replay_size, batch_size=batch_size,
        lr=lr, gamma=gamma, polyak=polyak, alpha=alpha, auto_alpha=auto_alpha,
        target_entropy=target_entropy, start_steps=start_steps, delay_update_steps=delay_update_steps,
        utd_ratio=utd_ratio, num_Q=num_Q, num_min=num_min, policy_update_delay=policy_update_delay,
        target_drop_rate=target_drop_rate, layer_norm=layer_norm, layer_norm_policy=layer_norm_policy,
        experience_group_size=experience_group_size, mask_dim=mask_dim,
        replay_mode=replay_mode, sumtree=sumtree,
        per_alpha=per_alpha, per_beta_start=per_beta_start, per_beta_end=per_beta_end,
        per_beta_anneal_steps=per_beta_anneal_steps,
    )

    # --- Dynamic PIToD controller (only in dynamic mode)
    controller = None
    h2_tracker = None
    if replay_mode == "dynamic_pitod":
        if h2_log:
            h2_tracker = H2Tracker(tag_step=h2_tag_step, tag_n_groups=h2_tag_n_groups)
        controller = DynamicPIToDController(
            agent=agent, registry=registry, sumtree=sumtree,
            k_refresh=k_refresh, b_refresh=b_refresh, m_strikes=m_strikes,
            epsilon_k=epsilon_k, pitod_alpha=pitod_alpha,
            n_samples_per_group=n_samples_per_group,
            warmup_steps=dynamic_warmup_steps,
            rng=np.random.RandomState(seed + 77),
            h2_tracker=h2_tracker,
        )

    wallclock_start = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):
        a = agent.get_exploration_action(o, env)
        o2, r, d, _ = env.step(a)

        if ((t >= (adversarial_reward_epoch * steps_per_epoch))
                and (t <= (adversarial_reward_epoch * steps_per_epoch + steps_per_epoch))):
            r = r * (-100.0)

        ep_len += 1
        d = False if ep_len == max_ep_len else d

        agent.store_data(o, a, r, o2, d)
        agent.train(logger)

        # --- Dynamic PIToD hooks ---
        if controller is not None:
            controller.on_new_transition(t)
            if (t + 1) % k_refresh == 0 and t >= dynamic_warmup_steps:
                stats = controller.refresh(t)
                logger.store(**{f"DynPIToD/{k}": v for k, v in stats.items()})

        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            test_agent(agent, test_env, max_ep_len, logger, n_eval=n_evals_per_epoch)
            if evaluate_bias:
                log_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff,
                               experience_cleansing=experience_cleansing,
                               dump_trajectory_for_demo=dump_trajectory_for_demo,
                               record_training_self_training_losses=record_training_self_training_losses,
                               influence_estimation_interval=influence_estimation_interval,
                               n_eval=n_eval)

            if reseed_each_epoch:
                seed_all(epoch)

            sps = (t + 1) / max(time.time() - wallclock_start, 1e-9)

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('SPS', sps)
            logger.log_tabular('ReplayMode', replay_mode)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1')
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if replay_mode == "dynamic_pitod":
                for key in ('DynPIToD/ScoreMean', 'DynPIToD/ScoreMin', 'DynPIToD/ScoreMax',
                            'DynPIToD/ScoreStd', 'DynPIToD/Epsilon', 'DynPIToD/MeanStrikes',
                            'DynPIToD/NumEvicted', 'DynPIToD/NumActive',
                            'DynPIToD/BufferActiveFrac', 'DynPIToD/GroupAgeMean',
                            'DynPIToD/GroupAgeMax', 'DynPIToD/NumRefreshed',
                            'DynPIToD/RefreshWallclock'):
                    try:
                        logger.log_tabular(key, average_only=True)
                    except Exception:
                        pass  # key may not have been populated yet if no refresh happened this epoch

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)

            logger.dump_tabular()
            sys.stdout.flush()

        if ((t % reset_interval) == 0) and (reset_interval >= 0):
            agent.reset()

    # --- end of run: dump H2 dataset if tracking was enabled ---
    if h2_tracker is not None and logger_kwargs.get('output_dir'):
        out_path = os.path.join(logger_kwargs['output_dir'], 'h2_dynamic_scores.bz2')
        h2_tracker.dump(out_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # --- inherited flags from main-TH.py ---
    parser.add_argument('-env', type=str, default='Hopper-v2')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=-1)
    parser.add_argument('-exp_name', type=str, default='redq_sac')
    parser.add_argument('-data_dir', type=str, default='./runs/')
    parser.add_argument('-info', type=str, default='DynPIToD')
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-target_drop_rate', type=float, default=0.0)
    parser.add_argument('-layer_norm', type=int, default=0, choices=[0, 1])
    parser.add_argument('-layer_norm_policy', type=int, default=0, choices=[0, 1])
    parser.add_argument('-num_q', type=int, default=2)
    parser.add_argument('-reset_interval', type=int, default=-1)
    parser.add_argument('-adversarial_reward_epoch', type=int, default=-999)
    parser.add_argument('-n_eval', type=int, default=10)
    parser.add_argument('-experience_group_size', type=int, default=5000)
    parser.add_argument('-hidden_sizes', type=int, nargs='+', default=[128, 128])
    parser.add_argument('-evaluate_bias', type=int, default=0, choices=[0, 1])
    parser.add_argument('-steps_per_epoch', type=int, default=5000)

    # --- new replay-mode flags ---
    parser.add_argument('--replay_mode', type=str, default='uniform',
                        choices=['uniform', 'per', 'static_pitod', 'dynamic_pitod'])
    parser.add_argument('--k_refresh', type=int, default=5000)
    parser.add_argument('--b_refresh', type=int, default=32)
    parser.add_argument('--m_strikes', type=int, default=3)
    parser.add_argument('--epsilon_k', type=float, default=1.0)
    parser.add_argument('--pitod_alpha', type=float, default=0.6)
    parser.add_argument('--n_samples_per_group', type=int, default=64)
    parser.add_argument('--dynamic_warmup_steps', type=int, default=5000)

    parser.add_argument('--per_alpha', type=float, default=0.6)
    parser.add_argument('--per_beta_start', type=float, default=0.4)
    parser.add_argument('--per_beta_end', type=float, default=1.0)
    parser.add_argument('--per_beta_anneal_steps', type=int, default=1_000_000)

    parser.add_argument('--h2_log', type=int, default=1, choices=[0, 1])
    parser.add_argument('--h2_tag_step', type=int, default=10_000)
    parser.add_argument('--h2_tag_n_groups', type=int, default=2)

    args = parser.parse_args()

    args.data_dir = args.data_dir + str(args.info) + '/'
    exp_name_full = f"{args.exp_name}_{args.env}_{args.replay_mode}"
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    dynamic_pitod(
        env_name=args.env, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, gpu_id=args.gpu_id,
        num_Q=args.num_q, layer_norm=bool(args.layer_norm),
        layer_norm_policy=bool(args.layer_norm_policy),
        target_drop_rate=args.target_drop_rate, reset_interval=args.reset_interval,
        adversarial_reward_epoch=args.adversarial_reward_epoch,
        n_eval=args.n_eval,
        experience_group_size=args.experience_group_size,
        hidden_sizes=tuple(args.hidden_sizes),
        evaluate_bias=bool(args.evaluate_bias),
        steps_per_epoch=args.steps_per_epoch,
        replay_mode=args.replay_mode,
        k_refresh=args.k_refresh,
        b_refresh=args.b_refresh,
        m_strikes=args.m_strikes,
        epsilon_k=args.epsilon_k,
        pitod_alpha=args.pitod_alpha,
        n_samples_per_group=args.n_samples_per_group,
        dynamic_warmup_steps=args.dynamic_warmup_steps,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        per_beta_anneal_steps=args.per_beta_anneal_steps,
        h2_log=bool(args.h2_log),
        h2_tag_step=args.h2_tag_step,
        h2_tag_n_groups=args.h2_tag_n_groups,
    )

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#torch.autograd.set_detect_anomaly(True)

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
#from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from env_pbeauty import PBeautyGame

from Simple_ReplayBuffer import SimpleReplayBuffer
from models import StochasiticNNConditionalPolicy,DeterminisiticNNPolicy,NNJointQFunction,NNQFunction
from mavb_ac import MAVBAC

from sampler import MASampler

from copy import deepcopy

def INI(modell):
    try:
        for m in modell.linear:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
    except:
        pass
    try:
        for m in modell.tmplinear:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
    except:
        pass

def get_agent(i,env):
    pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e4, agent_id=i)
    opponent_conditional_policy = StochasiticNNConditionalPolicy(env_spec=env.env_specs, agent_id=i, linearsth=[1,1,9])
    policy = DeterminisiticNNPolicy(env_spec=env.env_specs, agent_id=i)
    target_policy = DeterminisiticNNPolicy(env_spec=env.env_specs, agent_id=i)
    joint_qf = NNJointQFunction(env_spec=env.env_specs,agent_id=i,linearsth=[1,1,9])
    target_joint_qf = NNJointQFunction(env_spec=env.env_specs,agent_id=i,linearsth=[1,1,9])
    qf = NNQFunction(env_spec=env.env_specs,agent_id=i)
    INI(modell = opponent_conditional_policy)
    INI(modell = policy)
    INI(modell = target_policy)
    INI(modell = joint_qf)
    INI(modell = target_joint_qf)
    INI(modell = qf)
    agent = MAVBAC(
        agent_id=i,
        env=env,
        pool=pool,
        joint_qf=joint_qf,
        target_joint_qf=target_joint_qf,
        qf=qf,
        policy=policy,
        target_policy=target_policy,
        conditional_policy=opponent_conditional_policy,
        policy_lr=3e-4,
        qf_lr=3e-4,
        tau=0.01,
        value_n_particles=16,
        td_target_update_interval=5,
        discount=0.99,
        reward_scale=1
        )
    return agent


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    #envs_backup = make_vec_envs(args.env_name, args.seed, args.num_processes,args.gamma, args.log_dir, device, False)

    agent_num = 10

    envs = PBeautyGame(agent_num,reward_type="abs",p=1.1)

    #actor_1 = get_agent(env=envs,i=0)
    #actor_1._p_update()
    #actor_2 = get_agent(env=envs,i=1)
    actors = []

    for sth in range(agent_num):
        actors.append( get_agent(env=envs,i=sth) )
    
    batch_size = 64

    sampler = MASampler(agent_num=agent_num, joint=True, max_path_length=30, min_pool_size=100, batch_size=batch_size)
    sampler.initialize(envs,actors)

    for actor in actors:
        actor._target_update()
    
    initial_exploration_done = False

    for epoch in range(20000):
        
        #print(epoch)

        for t in range(1):
            #print("t",t)
            if not initial_exploration_done:
                if epoch >= 1000:
                    initial_exploration_done = True
            sampler.sample()
            if not initial_exploration_done:
                continue

            for j in range(1):
                batch_n = []
                recent_batch_n = []
                indices = None
                receent_indices = None
                for i, agent in enumerate(actors):
                    if i == 0:
                        batch = agent._pool.random_batch(batch_size)
                        indices = agent._pool.indices
                        receent_indices = list(range(agent._pool._top-batch_size, agent._pool._top))

                    batch_n.append(agent._pool.random_batch_by_indices(indices))
                    recent_batch_n.append(agent._pool.random_batch_by_indices(receent_indices))

                target_next_actions_n = []
                try:
                    for agent, batch in zip(actors, batch_n):
                        target_next_actions_n.append(agent._target_policy(torch.Tensor(batch['next_observations']).clone()))
                except:
                    pass
                
                opponent_actions_n = np.array([batch['actions'] for batch in batch_n])
                recent_opponent_actions_n = np.array([batch['actions'] for batch in recent_batch_n])

                recent_opponent_observations_n = []
                for batch in recent_batch_n:
                    recent_opponent_observations_n.append(batch['observations'])

                current_actions = [actors[i]._policy(torch.Tensor(batch_n[i]['next_observations']).clone())[0][0] for i in range(agent_num)]
                all_actions_k = []

                for i, agent in enumerate(actors):
                    #try:
                    #print(target_next_actions_n[i])
                    batch_n[i]['next_actions'] = deepcopy(target_next_actions_n[i].detach())
                    
                    #except:
                        #pass
                    batch_n[i]['opponent_actions'] = np.reshape(np.delete(deepcopy(opponent_actions_n), i, 0), (-1, agent._opponent_action_dim))
                    
                    agent._do_training(iteration=t + epoch * 1000, batch=batch_n[i], annealing=0.5)

    
    """

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.reset().shape, envs.action_spaces,
                              actor_critic.recurrent_hidden_state_size)

    #rollouts_backup = RolloutStorage(args.num_steps, args.num_processes,
    #                                envs_backup.observations.shape,envs_backup.action_space,
    #                                actor_critic.recurrent_hidden_state_size)

    obs = torch.from_numpy(envs.reset())
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)#10个元素的固定大小队列

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():

                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    #print(info['episode']['r'])
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            # masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            # bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)]
    """


if __name__ == "__main__":
    main()

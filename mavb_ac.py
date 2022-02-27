import numpy as np
import torch
import torch.nn as nn

from kernel import adaptive_isotropic_gaussian_kernel

from einops import repeat

from copy import deepcopy

EPS = 1e-6

from torch.utils.tensorboard import SummaryWriter

class MAVBAC():
    def __init__(
        self,
        agent_id,
        env,
        pool,
        joint_qf,
        target_joint_qf,
        qf,
        policy,
        target_policy,
        conditional_policy,
        policy_lr=1E-3,
        qf_lr=1E-3,
        tau=0.01,
        value_n_particles=16,
        td_target_update_interval=1,
        discount=0.99,
        reward_scale=1,
    ):
        super(MAVBAC, self).__init__()
        
        self._env = env
        self._pool = pool
        self.qf = qf
        self.joint_qf = joint_qf
        self.target_joint_qf = target_joint_qf
        self._policy = policy
        self._target_policy = target_policy
        self._conditional_policy = conditional_policy

        self._agent_id = agent_id

        self._tau = tau
        self._qf_lr = qf_lr
        self._policy_lr = policy_lr
        self._discount = discount
        self._reward_scale = reward_scale

        self._value_n_particles = value_n_particles
        self._qf_target_update_interval = td_target_update_interval

        self._observation_dim = self._env.observation_spaces[self._agent_id].flat_dim
        self._action_dim = self._env.action_spaces[self._agent_id].flat_dim

        self._opponent_action_dim = self._env.action_spaces.opponent_flat_dim(self._agent_id)

        self._training_ops = []
        self._target_ops = []

        self._annealing = 0.5

        self._kernel_n_particles = 32
        self._kernel_update_ratio = 0.5

        self.writer = SummaryWriter('runs/agent_'+str(agent_id))

    def _q_update(self):
        opponent_target_actions = (
            2*torch.rand(1,self._value_n_particles,self._opponent_action_dim)-1
        ).expand(64,self._value_n_particles,self._opponent_action_dim)

        q_value_targets = self.target_joint_qf(
            (self._next_observations[:,None,:],self._next_actions[:,None,:],opponent_target_actions)
        )

        self._q_values = self.joint_qf(
            (self._observations,self._actions,self._opponent_actions)
        )
        
        next_value = self._annealing * torch.logsumexp((q_value_targets / self._annealing),dim=1)#,dim=0)

        sth = torch.from_numpy(np.array(self._value_n_particles))

        next_value -= torch.log(sth)
        next_value += (self._opponent_action_dim) * np.log(2)

        ys = self._rewards + (1 - self._terminals) * 0.99 * next_value
        ys = ys.detach().clone()#.detach()

        bellman_residual = 0.5 * torch.mean((ys - self._q_values)**2)

        self.bellman_residual = bellman_residual

        optimizer = torch.optim.Adam(self.joint_qf.parameters(),self._qf_lr)

        optimizer.zero_grad()
        bellman_residual.backward()
        optimizer.step()

    def _p_update(self):
        self_actions = self._policy(self._observations)

        opponent_target_actions = self._conditional_policy(
            inputs = (self._observations,self._actions),
            observations = self._observations,
            n_action_samples = self._value_n_particles
        )

        q_targets = self.joint_qf(
            (self._next_observations[:,None,:],self_actions[:,None,:],opponent_target_actions)
        )

        q_targets = self._annealing * torch.logsumexp(q_targets / self._annealing,dim=1).clone()
        q_targets -= torch.log(torch.Tensor([self._value_n_particles]))
        q_targets += (self._opponent_action_dim) * np.log(2)
        pg_loss = -torch.mean(q_targets)
        
        optimizer = torch.optim.Adam(self._policy.parameters(),self._policy_lr)

        optimizer.zero_grad()
        pg_loss.backward()
        optimizer.step()
    
    def _conditional_policy_svgd_update(self):
        actions = self._conditional_policy(
            inputs = (self._observations,self._actions),
            observations = self._observations,
            n_action_samples = self._kernel_n_particles
        )

        n_updated_actions = int(
            self._kernel_n_particles * self._kernel_update_ratio
        )
        n_fixed_actions = self._kernel_n_particles - n_updated_actions

        fixed_actions,updated_actions = torch.split(actions,[n_fixed_actions, n_updated_actions], dim=1)
        fixed_actions = fixed_actions.detach().clone()
        fixed_actions = torch.autograd.Variable(fixed_actions,requires_grad=True)

        svgd_target_values = self.joint_qf(
            (self._observations[:,None,:],self._actions[:,None,:],fixed_actions)#FIXED)#fixed_actions)
        )

        baseline_ind_q = self.qf(torch.cat((self._observations, self._actions),dim=1))

        baseline_ind_q = torch.reshape(baseline_ind_q, [-1,1])
        baseline_ind_q = repeat(baseline_ind_q,'h w -> h (repeat w)',repeat = n_fixed_actions)

        svgd_target_values = (svgd_target_values - baseline_ind_q) / self._annealing

        squash_correction = torch.sum(
            torch.log(1 - fixed_actions**2 + EPS),#fixed_actions**2 + EPS),
            axis = -1
        )
        log_p = svgd_target_values + squash_correction

        grad_log_p = torch.autograd.grad(
            outputs = log_p,
            inputs = fixed_actions,
            grad_outputs=torch.ones(log_p.shape),
        )[0]

        grad_log_p = grad_log_p[:,:,None,:]
        grad_log_p = grad_log_p.detach().clone()

        kernel_dict = adaptive_isotropic_gaussian_kernel(xs=fixed_actions, ys=updated_actions)

        kappa = kernel_dict["output"].unsqueeze(3)

        action_gradients = torch.mean(
            kappa * grad_log_p + kernel_dict["gradient"], dim=1)

        gradients = torch.autograd.grad(
            outputs = updated_actions,
            inputs = self._conditional_policy.parameters(),
            grad_outputs=action_gradients,
        )

        tmp = torch.Tensor([0])                                                        
        for w, g in zip(self._conditional_policy.parameters(), gradients):
            tmp = tmp + torch.sum(w * g.detach().clone())

        surrogate_loss = -torch.sum(tmp)

        optimizer = torch.optim.Adam(self._conditional_policy.parameters(),self._policy_lr)
        optimizer.zero_grad()
        surrogate_loss.backward()

        optimizer.step()

    def _target_update(self):
        source_q_params = self.joint_qf.parameters()
        target_q_params = self.target_joint_qf.parameters()
        source_p_params = self._policy.parameters()
        target_p_params = self._target_policy.parameters()

        for target, source in zip(target_q_params, source_q_params):
            target.data = (1 - self._tau) * target.data.detach().clone() + self._tau * source.data.detach().clone()
        
        for target, source in zip(target_p_params, source_p_params):
            target.data = (1 - self._tau) * target.data.detach().clone() + self._tau * source.data.detach().clone()
    
    def get_feed_dict(self,batch,annealing):
        self._observations = torch.Tensor(batch['observations'])
        self._actions = torch.Tensor(batch['actions'])
        self._opponent_actions = torch.Tensor(batch['opponent_actions'])
        self._next_actions = torch.Tensor(batch['next_actions'])
        self._next_observations = torch.Tensor(batch['next_observations'])
        self._rewards = torch.Tensor(batch['rewards'])
        self._terminals = torch.Tensor(batch['terminals'])
        self._annealing = annealing

    def _do_training(self,iteration,batch,annealing=1.):
        self.get_feed_dict(batch,annealing)
        self._q_update()
        self._p_update()
        self._conditional_policy_svgd_update()
        if iteration % self._qf_target_update_interval == 0:
            self._target_update()
        if iteration % 1 == 0:
            self.writer.add_scalar('avg',torch.mean(self._q_values),global_step=iteration)
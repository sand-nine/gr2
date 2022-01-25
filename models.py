import numpy as np
import torch
import torch.nn as nn

class StochasiticNNConditionalPolicy(nn.Module):
    def __init__(self, env_spec, agent_id, hidden_size=100,linearsth=[]):
        super(StochasiticNNConditionalPolicy,self).__init__()

        self._action_dim = env_spec.action_space[agent_id].flat_dim
        self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)

        self.linear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,self._opponent_action_dim),
            nn.Tanh()
        )

        self.action_linear = nn.Linear(linearsth[0],100)
        self.obs_linear = nn.Linear(linearsth[1],100)
        self.latent_linear = nn.Linear(linearsth[2],100)

        self.tmplinear = [
            self.action_linear,
            self.obs_linear,
            self.latent_linear
        ]
        
    def forward(self,inputs,observations,n_action_samples=1):

        n_state_samples = observations.shape[0]

        #latent = torch.rand(observations.shape[0],self._opponent_action_dim)

        inputs_tmp = ()

        if n_action_samples > 1:
            for sth in inputs:
                inputs_tmp = inputs_tmp+(sth[:,None,:],)
            inputs = inputs_tmp
            latent_shape = (n_state_samples, n_action_samples,
                            self._opponent_action_dim)
        else:
            latent_shape = (n_state_samples, self._opponent_action_dim)

        latent = torch.rand(latent_shape)

        out = torch.Tensor([0])
        sth = (latent,)
        
        for j,input_tensor in enumerate(inputs + sth):
            tmp = self.tmplinear[j](input_tensor)
            out = out.expand(tmp.size()).clone()
            out += tmp
        out = self.linear(out)
        return out

class DeterminisiticNNPolicy(nn.Module):
    def __init__(self, env_spec, agent_id, hidden_size=100):
        super(DeterminisiticNNPolicy,self).__init__()

        self._action_dim = env_spec.action_space[agent_id].flat_dim
        self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)

        self.fc0 = nn.Linear(self._observation_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, self._action_dim)
        self.tanh = nn.Tanh()

    def forward(self,x):
        out = self.fc0(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out

class NNJointQFunction(nn.Module):
    def __init__(self,env_spec,hidden_size=100,agent_id=None,linearsth=[]):
        super(NNJointQFunction,self).__init__()

        self._action_dim = env_spec.action_space[agent_id].flat_dim
        self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)

        self.linear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(inplace=True),
            #nn.Linear(hidden_size,self._opponent_action_dim)
            nn.Linear(hidden_size,self._action_dim)
        )

        self.action_linear = nn.Linear(linearsth[0],100)
        self.obs_linear = nn.Linear(linearsth[1],100)
        self.tmp_linear = nn.Linear(linearsth[2],100)

        self.tmplinear = [
            self.action_linear,
            self.obs_linear,
            self.tmp_linear
        ]

    def forward(self, inputs):
        out = torch.Tensor([0])
        for j,input_tensor in enumerate(inputs):
            tmp = self.tmplinear[j](input_tensor)#.detach().numpy()
            
            try:
                out = out.expand(tmp.size()).clone()
            except:
                tmp = tmp.expand(out.size()).clone()
            out += tmp
        out = self.linear(out)
        return out[...,0]

class NNQFunction(nn.Module):
    def __init__(self,env_spec,hidden_size=100,agent_id=None):
        super(NNQFunction,self).__init__()

        self._action_dim = env_spec.action_space[agent_id].flat_dim
        self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)

        self.linear = nn.Sequential(
            nn.Linear(self._observation_dim + self._action_dim , hidden_size ),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,self._opponent_action_dim)
        )
    
    def forward(self, x):
        out = self.linear(x)
        return out[...,0]

"""
class StochasiticNNConditionalPolicy(nn.Module):
    def __init__(self,env_spec,hidden_size=100,agent_id=None):
        super(StochasiticNNConditionalPolicy,self).__init__()

        self._action_dim = env_spec.action_space[agent_id].flat_dim
        self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)

        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,self._opponent_action_dim),
            nn.Tanh()
        )
    def forward(self, inputs):
        out = torch.Tensor([0])
        for j,input_tensor in enumerate(inputs):
            tmplinear = nn.Sequential(nn.Linear(input_tensor.shape[-1], 100 ))
            tmp = tmplinear(input_tensor)
            out = out.expand(tmp.size()).clone()
            out += tmp
        out = self.linear(out)
        return out

class DeterminisiticNNPolicy(nn.Module):
    def __init__(self, env_spec, agent_id, hidden_size=100):
        super(DeterminisiticNNPolicy,self).__init__()

        self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)

        self.fc0 = nn.Linear(self._observation_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self._opponent_action_dim)
        self.tanh = nn.Tanh()

    def forward(self,x):
        out = self.fc0(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out

class NNJointQFunction2(nn.Module):
    def __init__(self,env_spec,hidden_size=100,agent_id=None):
        super(NNJointQFunction2,self).__init__()

        self._action_dim = env_spec.action_space[agent_id].flat_dim
        self._observation_dim = env_spec.observation_space[agent_id].flat_dim
        self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)

        self.linear = nn.Sequential(
            nn.Linear(self._observation_dim + self._action_dim*2 , hidden_size ),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,self._opponent_action_dim)
        )
    
    def forward(self, x):
        out = self.linear(x)
        return out
"""
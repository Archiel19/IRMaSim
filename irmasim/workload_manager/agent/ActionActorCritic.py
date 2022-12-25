import numpy as np
from scipy.signal import lfilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from irmasim.Options import Options
from irmasim.workload_manager.agent.Agent import Agent


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class ActionActorCritic(Agent):

    def __init__(self, actions_size: int, observation_size: tuple) -> None:
        super(ActionActorCritic, self).__init__()
        self.actor = ActionActor(actions_size, observation_size[1])
        self.critic = ActionCritic(actions_size, observation_size[1])
        options = Options().get()
        self.buffer = PPOBuffer(observation_size, (), options['trajectory_length'])

    def decide(self, observation: torch.Tensor) -> tuple:
        with torch.no_grad():
            pi, v = self.forward(observation)
            a, logp_a = self.get_action(pi)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def get_action(self, dist: Categorical) -> tuple:
        action = dist.sample()
        return action, dist.log_prob(action)

    def forward(self, observation: torch.Tensor) -> tuple:
        pi, _ = self.actor.forward(observation)
        value = self.critic.forward(observation)
        return pi, value

    def store_values(self, obs, act, rew, val, logp):
        self.buffer.store(obs, act, rew, val, logp)

    def fixed_reward(self, rew: float):
        self.last_rew = rew
        self.store_values()

    def store_values(self):
        self.buffer.store(self.last_rew, self.last_v, self.last_logp)

    def finish_trajectory(self, rew: float):
        self.buffer.finish_path(rew)

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])


class ActionActor(nn.Module):
    clip_ratio = 0.2

    def __init__(self, actions_size: int, observation_size: int):
        super(ActionActor, self).__init__()
        self.input = nn.Linear(observation_size, 32)
        self.actor_hidden_0 = nn.Linear(32, 16)
        self.actor_hidden_1 = nn.Linear(16, 8)
        self.actor_output = nn.Linear(8, 1)

    def forward(self, observation: torch.Tensor) -> tuple:
        mask = torch.where(observation.sum(dim=-1) != 0.0, 1.0, 0.0)
        return self.forward_action(observation, mask)

    def forward_action(self, observation: torch.Tensor, mask: torch.Tensor) -> tuple:
        out_0 = F.leaky_relu(self.input(observation))
        out_1 = F.leaky_relu(self.actor_hidden_0(out_0))
        out_2 = F.leaky_relu(self.actor_hidden_1(out_1))
        out_3 = torch.squeeze(self.actor_output(out_2), dim=-1)
        out = out_3 + (mask - 1) * 1e6
        logp_all = F.log_softmax(out, dim=1)
        return logp_all, out
        # logp = torch.sum(F.one_hot(a) * logp_all, dim=1)
        # logp_pi = torch.sum(F.one_hot(pi) * logp_all, dim=1)
        # return torch.squeeze(out)  # Col of scores to row

    def loss(self, adv_ph, logp_old_ph) -> torch.Tensor:
        # ratio = torch.exp(logp - logp_old_ph)
        min_adv = torch.where(adv_ph > 0, (1 + self.clip_ratio) * adv_ph, (1 - self.clip_ratio) * adv_ph)
        # pi_loss = torch.mean(torch.minimum(ratio * adv_ph, min_adv))
        pi_loss = torch.mean(min_adv)
        return pi_loss


class ActionCritic(nn.Module):
    def __init__(self, actions_size: int, observation_size: int):
        super(ActionCritic, self).__init__()
        self.input = nn.Linear(observation_size, 32)
        self.critic_hidden_0_0 = nn.Linear(32, 16)
        self.critic_hidden_0_1 = nn.Linear(16, 8)
        self.critic_hidden_0_2 = nn.Linear(8, 1)
        self.critic_hidden_1_0 = nn.Linear(actions_size, 64)
        self.critic_hidden_1_1 = nn.Linear(64, 32)
        self.critic_hidden_1_2 = nn.Linear(32, 8)
        self.critic_output = nn.Linear(8, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        out_0_0 = F.leaky_relu(self.input(observation))
        out_0_1 = F.leaky_relu(self.critic_hidden_0_0(out_0_0))
        out_0_2 = F.leaky_relu(self.critic_hidden_0_1(out_0_1))
        out_0_3 = F.leaky_relu(self.critic_hidden_0_2(out_0_2))
        out_1 = torch.squeeze(out_0_3)
        out_1_1 = F.leaky_relu(self.critic_hidden_1_0(out_1))
        out_1_2 = F.leaky_relu(self.critic_hidden_1_1(out_1_1))
        out_1_3 = F.leaky_relu(self.critic_hidden_1_2(out_1_2))
        return self.critic_output(out_1_3)

    def loss(self, data: dict) -> torch.Tensor:
        obs, ret = data['obs'], data['ret']
        return ((ret - self.forward(obs)) ** 2).mean()


class PPOBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.rewards = []

    def store(self, obs, act, rew, val, logp) -> None:
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def drop_last_values(self):
        self.ptr -= 1

    def finish_path(self, last_val) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        self.rewards = rews

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self) -> dict:
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPOTrainer:

    def __init__(self, agent: ActionActorCritic, optim_pi, optim_v, train_pi_iters=80, train_v_iters=80, target_kl=0.01):
        self.agent = agent
        self.optim_pi = optim_pi
        self.optim_v = optim_v
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl

    def update(self) -> np.ndarray:
        data = self.agent.buffer.get()
        losses = np.zeros((self.train_pi_iters, 2), dtype=np.float32)

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.optim_pi.zero_grad()
            loss_pi, kl = self.agent.actor.loss(data)
            losses[i][0] = loss_pi.item()
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self.optim_pi.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.optim_v.zero_grad()
            loss_v = self.agent.critic.loss(data)
            losses[i][1] = loss_v.item()
            loss_v.backward()
            self.optim_v.step()

        return np.mean(losses, axis=0)


# TODO Remove
if __name__ == "__main__":
    JOBS = 3
    NODES = 4
    ac = ActionActorCritic(JOBS * NODES + 4, 5)
    obs = np.random.uniform(0.0, 1.0, (JOBS * NODES, 5))
    obs = np.pad(obs, [(0, 4), (0, 0)])
    print('INPUT: ', obs)
    print('--- End input')
    action = ac.decide(obs)
    print(action)
    print('-----------------')
    t = torch.Tensor(obs)
    r = torch.Tensor(obs)
    s = torch.Tensor(obs)
    u = (t + r * s)

    mask = np.array(r.detach().numpy().sum(axis=1) != 0.0, dtype=np.float32)
    mask2 = torch.where(r.sum(dim=1) != 0.0, 1.0, 0.0)
    #out = ac.actor.forward_action(r, torch.Tensor(mask))


    print('-----------------')
    #print(out)
    print(mask)
    print(mask2)
    #print(out * torch.Tensor(mask))



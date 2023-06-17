import numpy as np
from typing import Tuple, Any

from irmasim.Options import Options
from scipy.signal import lfilter
import os.path as path
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim.swa_utils import AveragedModel, SWALR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BUFFER_SIZE_MPLIER = 5


def init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()

class EnergyActorCritic:
    def __init__(self, observation_size: tuple, load_agent: bool = True):
        wait_action = Options().get()['workload_manager']['wait_action']
        buffer_size = Options().get()['nbtrajectories'] * Options().get()['trajectory_length']
        if wait_action:
            buffer_size *= BUFFER_SIZE_MPLIER
        self.buffer = PPOBuffer(observation_size, buffer_size)
        self.actor = EnergyActorCritic.Actor(observation_size, wait_action)
        self.critic = EnergyActorCritic.Critic(observation_size)

        self.agent_options = Options().get()['workload_manager']['agent']
        self.lam = self.agent_options['lambda']
        self.gamma = self.agent_options['gamma']
        self.swa_on = False
        if 'swa' in self.agent_options:
            self.swa_on = self.agent_options['swa']

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.agent_options['lr_pi']))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(self.agent_options['lr_v']))

        # For Stochastic Weight Averaging
        lr_pi = float(self.agent_options['lr_pi'])
        iters = int(self.agent_options['train_iters'])
        self.SWA_START_PC = 0.75
        self.swa_start_epoch = int(self.SWA_START_PC * iters)
        self.actor_swa_model = None
        self.actor_swa_scheduler = SWALR(self.actor_optimizer,
                                         anneal_strategy="linear",
                                         anneal_epochs=int(0.15 * iters),
                                         swa_lr=lr_pi)

        if 'input_model' in self.agent_options \
                and path.isfile(self.agent_options['input_model']) and load_agent:
            in_model = self.agent_options['input_model']
            print(f'Reading model to {in_model}')
            checkpoint = torch.load(in_model)
            self.load_state_dict(checkpoint['model_state_dict'])
            checkpoint['optimizer_state_dict']['pi']['param_groups'][0]['lr'] = float(self.agent_options['lr_pi'])
            checkpoint['optimizer_state_dict']['v']['param_groups'][0]['lr'] = float(self.agent_options['lr_v'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['pi'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['v'])



    def state_dict(self):
        return {
            'actor': self.actor_swa_model.state_dict() if self.actor_swa_model else self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        self.actor.load_state_dict(state_dict['actor'], strict=False)
        self.critic.load_state_dict(state_dict['critic'])

    def store(self, observation, action, value, logp, reward=0):
        self.buffer.store(observation, action, value, logp, reward)

    def decide(self, observation: torch.Tensor) -> tuple:
        with torch.no_grad():  # Grad will be used when computing losses, but not when generating training data
            action, logp_action, _ = self.actor.forward(observation.to(DEVICE))
            value = self.critic.forward(observation.to(DEVICE))
        return action.item(), value.item(), logp_action.item()

    @property
    def total_rewards(self):
        return self.buffer.total_rewards

    def reward_last_action(self, reward):
        self.buffer.store_past_reward(reward)

    def on_end_trajectory(self, last_reward):
        self.buffer.on_end_trajectory(self.gamma, self.lam, last_reward)

    def training_step(self):
        """
        Training step, executed after each simulation.

        Uses Adam optimizers for both actor and critic. The collected data is
        divided into minibatches for training.
        Implements a dynamic learning rate and stochastic weight averaging.
        """
        num_samples = self.buffer.num_samples
        minibatch_size = self.agent_options['minibatch_size']
        train_iters = self.agent_options['train_iters']
        h_factor = self.agent_options['h_factor']
        epsilon = self.agent_options['clipping_factor']

        # Normalize advantages, reset buffer and collect data
        training_data = self.buffer.collect()
        losses = np.zeros((train_iters, 2), dtype=np.float32)

        def next_minibatch():
            for i in range(0, num_samples, minibatch_size):
                m = {}
                for k in self.buffer.dict.keys():
                    m[k] = training_data[k][i:i+minibatch_size]
                yield m

        for epoch in range(train_iters):
            for minibatch in next_minibatch():

                # Train actor
                self.actor_optimizer.zero_grad()
                temp_loss = self.actor.loss(minibatch, epsilon)
                _, _, entropy = self.actor.forward(minibatch['obs'])
                loss_pi = temp_loss - (h_factor * entropy)
                losses[epoch][0] = loss_pi.item()
                loss_pi.backward()
                self.actor_optimizer.step()

                # Train critic
                self.critic_optimizer.zero_grad()
                loss_v = self.critic.loss(minibatch)
                losses[epoch][1] = loss_v.item()
                loss_v.backward()
                self.critic_optimizer.step()

            if self.swa_on and epoch > self.swa_start_epoch:
                if self.actor_swa_model:
                    self.actor_swa_model.update_parameters(self.actor)
                    self.actor_swa_scheduler.step()
                else:
                    self.actor_swa_model = AveragedModel(self.actor, device=DEVICE)
        return np.mean(losses, axis=0)

    class Actor(nn.Module):
        def __init__(self, observation_size, wait_action):
            super(EnergyActorCritic.Actor, self).__init__()
            self.wait_action = wait_action
            self.model = nn.Sequential(
                nn.Linear(observation_size[1], 16), nn.SELU(),
                nn.Linear(16, 16, device=DEVICE), nn.SELU(),
                nn.Linear(16, 8, device=DEVICE), nn.SELU(),
                nn.Linear(8, 8, device=DEVICE), nn.SELU(),
                nn.Linear(8, 4, device=DEVICE), nn.SELU(),
                nn.Linear(4, 4, device=DEVICE), nn.SELU(),
                nn.Linear(4, 1, device=DEVICE), nn.SELU()
            ).to(DEVICE)
            self.apply(init_weights)

        def forward(self, observation: torch.Tensor, act=None) -> Tuple[Any, Any, Any]:
            mask = torch.where(observation.sum(dim=-1) != 0.0, 1.0, 0.0)
            if self.wait_action:
                mask[-1] = 1.0
            out = self.model(observation)
            out = torch.squeeze(out, dim=-1)
            out = out + (mask - 1.0) * 1e6
            pi = Categorical(logits=out)
            action = pi.sample() if act is None else act
            return action, pi.log_prob(action), torch.mean(pi.entropy())

        def loss(self, minibatch: dict, epsilon=0.2):
            obs = minibatch['obs']
            act = minibatch['act']
            adv = minibatch['adv']
            logp_old = minibatch['logp']

            _, logp, _ = self.forward(obs, act)
            ratio = torch.exp(logp - logp_old)
            clipped = torch.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
            loss = -torch.mean(torch.minimum(ratio * adv, clipped))

            return loss

    class Critic(nn.Module):
        def __init__(self, observation_size):
            super(EnergyActorCritic.Critic, self).__init__()
            self.input = nn.Linear(observation_size[1], 32, device=DEVICE)
            self.hidden_0_0 = nn.Linear(32, 16, device=DEVICE)
            self.hidden_0_1 = nn.Linear(16, 8, device=DEVICE)
            self.hidden_0_2 = nn.Linear(8, 4, device=DEVICE)
            self.hidden_0_3 = nn.Linear(4, 1, device=DEVICE)

            self.hidden_1_0 = nn.Linear(observation_size[0], 128, device=DEVICE)
            self.hidden_1_1 = nn.Linear(128, 64, device=DEVICE)
            self.hidden_1_2 = nn.Linear(64, 32, device=DEVICE)
            self.hidden_1_3 = nn.Linear(32, 16, device=DEVICE)
            self.hidden_1_4 = nn.Linear(16, 8, device=DEVICE)
            self.output = nn.Linear(8, 1, device=DEVICE)
            self.apply(init_weights)

        def forward(self, observation: torch.Tensor) -> torch.Tensor:

            # First phase output: dim = (num_batches, all job-node combinations (actions_size))
            out_0_0 = nn.functional.selu(self.input(observation))
            out_0_1 = nn.functional.selu(self.hidden_0_0(out_0_0))
            out_0_2 = nn.functional.selu(self.hidden_0_1(out_0_1))
            out_0_3 = nn.functional.selu(self.hidden_0_2(out_0_2))
            out_0_4 = nn.functional.selu(self.hidden_0_3(out_0_3))
            out_1 = torch.squeeze(out_0_4)

            out_1_1 = nn.functional.selu(self.hidden_1_0(out_1))
            out_1_2 = nn.functional.selu(self.hidden_1_1(out_1_1))
            out_1_3 = nn.functional.selu(self.hidden_1_2(out_1_2))
            out_1_4 = nn.functional.selu(self.hidden_1_3(out_1_3))
            out_1_5 = nn.functional.selu(self.hidden_1_4(out_1_4))
            out = self.output(out_1_5)
            return out

        def loss(self, minibatch: dict):
            obs = minibatch['obs']
            ret = minibatch['ret']
            if ret.shape[0] != 1:
                ret = ret.unsqueeze(1)
            loss = torch.nn.MSELoss()
            return loss(self.forward(obs), ret).mean()



class PPOBuffer:
    def __init__(self, observation_size, buffer_size):
        self.start_ptr = 0
        self.curr_ptr = 0
        self.max_size = buffer_size
        self.total_rewards = 0

        self.obs = np.zeros((buffer_size, *observation_size))
        self.act = np.zeros(buffer_size)
        self.rew = np.zeros(buffer_size)
        self.val = np.zeros(buffer_size)
        self.logp = np.zeros(buffer_size)

        self.adv = np.zeros(buffer_size)
        self.ret = np.zeros(buffer_size)

        self.dict = {'obs': self.obs, 'act': self.act, 'rew': self.rew, 'val': self.val,
                     'logp': self.logp, 'adv': self.adv, 'ret': self.ret}

    @property
    def num_samples(self):
        return self.curr_ptr

    def store(self, observation, action, value, logp, reward) -> None:
        assert self.curr_ptr < self.max_size
        self.obs[self.curr_ptr] = observation
        self.act[self.curr_ptr] = action
        self.rew[self.curr_ptr] = reward
        self.val[self.curr_ptr] = value
        self.logp[self.curr_ptr] = logp
        self.curr_ptr += 1

    def store_past_reward(self, reward) -> None:
        if 0 < self.curr_ptr <= self.max_size:
            self.rew[self.curr_ptr - 1] = reward

    def on_end_trajectory(self, gamma, lam, last_reward) -> None:
        self.store_past_reward(last_reward)
        trajectory_slice = slice(self.start_ptr, self.curr_ptr)

        # Compute advantages and rewards-to-go
        rewards = self.rew[trajectory_slice]
        values = self.val[trajectory_slice]
        self.total_rewards = np.sum(rewards)
        values = np.append(values, 0) # This acts as our mask to avoid taking into account the last value

        def discounted_sum(array, discount):
            y = lfilter([1], [1, -discount], x=array[::-1])
            return y[::-1]

        deltas = rewards + (gamma * values[1:]) - values[:-1]
        advantages = discounted_sum(deltas, gamma * lam)
        expected_returns = discounted_sum(rewards, gamma)

        self.adv[trajectory_slice] = advantages
        self.ret[trajectory_slice] = expected_returns

        # Advance
        self.start_ptr = self.curr_ptr

    def collect(self) -> dict:
        """
        Reset and normalize advantages

        Returns collected data
        """

        # Normalize advantages
        adv_mean = np.mean(self.adv)
        adv_std = np.std(self.adv)
        self.adv = (self.adv - adv_mean) / adv_std

        # Create list of samples
        sample_dict = {}
        for k, v in self.dict.items():
            sample_dict[k] = torch.as_tensor(v[:self.num_samples], dtype=torch.float32, device=DEVICE)
        return sample_dict
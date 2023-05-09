import numpy as np
from typing import Tuple, Any

from irmasim.Options import Options
from scipy.signal import lfilter
import os.path as path
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import LinearLR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EnergyActorCritic:
    def __init__(self, observation_size: tuple, load_agent: bool = True):
        wait_action = Options().get()['workload_manager']['wait_action']
        self.actor = EnergyActorCritic.Actor(observation_size, wait_action)
        self.critic = EnergyActorCritic.Critic(observation_size)
        self.buffer = EnergyActorCritic.PPOBuffer()

        self.agent_options = Options().get()['workload_manager']['agent']
        self.lam = self.agent_options['lambda']
        self.gamma = self.agent_options['gamma']

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.agent_options['lr_pi']))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(self.agent_options['lr_v']))

        if 'input_model' in self.agent_options \
                and path.isfile(self.agent_options['input_model']) \
                and load_agent:
            in_model = self.agent_options['input_model']
            print(f'Reading model to {in_model}')
            checkpoint = torch.load(in_model)
            self.load_state_dict(checkpoint['model_state_dict'])
            checkpoint['optimizer_state_dict']['pi']['param_groups'][0]['lr'] = float(self.agent_options['lr_pi'])
            checkpoint['optimizer_state_dict']['v']['param_groups'][0]['lr'] = float(self.agent_options['lr_v'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['pi'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['v'])

        # For learning rate scheduling and Stochastic Weight Averaging
        lr_pi = float(self.agent_options['lr_pi'])
        iters = int(self.agent_options['train_iters'])
        self.SWA_START_PC = 0.9
        self.swa_start_epoch = int(self.SWA_START_PC * iters)
        self.actor_swa_model = None
        self.actor_scheduler = LinearLR(self.actor_optimizer,
                                        start_factor=lr_pi,
                                        end_factor=0.1 * lr_pi,
                                        total_iters=self.swa_start_epoch)
        self.actor_swa_scheduler = SWALR(self.actor_optimizer,
                                         anneal_strategy="linear",
                                         anneal_epochs=int(0.05 * iters),
                                         swa_lr=0.1 * lr_pi)


    def state_dict(self):
        return {
            'actor': self.actor_swa_model.state_dict() if self.actor_swa_model else self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        self.actor.load_state_dict(state_dict['actor'], strict=False)
        self.critic.load_state_dict(state_dict['critic'])

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
        Training step, executed after each simulation

        Based on the code of the guy who did his master's thesis on PPO,
        and on ActionActorCritic
        """
        num_samples = self.buffer.num_samples()
        minibatch_size = self.agent_options['minibatch_size']
        train_iters = self.agent_options['train_iters']
        h_factor = self.agent_options['h_factor']
        epsilon = self.agent_options['clipping_factor']
        max_kl = self.agent_options['max_kl']

        # Normalize advantages, reset buffer and collect data
        training_data = self.buffer.on_end_simulation()
        losses = np.zeros((train_iters, 2), dtype=np.float32)

        def next_minibatch():
            for i in range(0, num_samples, minibatch_size):
                m = {}
                for k in self.buffer.dict.keys():
                    m[k] = training_data[k][i:i+minibatch_size]
                yield m

        for epoch in range(train_iters):
            kl = 0
            for minibatch in next_minibatch():
                # Actor
                self.actor_optimizer.zero_grad()
                temp_loss, kl = self.actor.loss(minibatch, epsilon)
                _, _, entropy = self.actor.forward(minibatch['obs'])
                loss_pi = temp_loss - (h_factor * entropy)
                losses[epoch][0] = loss_pi.item()
                loss_pi.backward()
                self.actor_optimizer.step()

                # Critic
                self.critic_optimizer.zero_grad()
                loss_v = self.critic.loss(minibatch)
                losses[epoch][1] = loss_v.item()
                loss_v.backward()
                self.critic_optimizer.step()

            if epoch > self.swa_start_epoch:
                if self.actor_swa_model:
                    self.actor_swa_model.update_parameters(self.actor)
                    self.actor_swa_scheduler.step()
                else:
                    self.actor_swa_model = AveragedModel(self.actor, device=DEVICE)
            else:
                self.actor_scheduler.step()
            if kl > 1.5 * max_kl:
                break
        return np.mean(losses, axis=0)

    class PPOBuffer:
        def __init__(self):
            self.start_ptr = 0
            self.curr_ptr = 0
            self.total_rewards = 0

            self.obs = []
            self.act = []
            self.rew = []
            self.val = []
            self.logp = []
            self.dict = {'obs': self.obs, 'act': self.act, 'rew': self.rew, 'val': self.val, 'logp': self.logp}

            # Computed at the end of each trajectory making use of numpy
            # TODO: Small problem: have to be manually assigned to the dict at the end of each simulation because
            # TODO: np.append() is used, changing the reference.
            self.adv = np.empty(0)
            self.ret = np.empty(0)

        def num_samples(self):
            return len(self.act)  # For example

        def _reset(self):
            self.start_ptr = 0
            self.curr_ptr = 0

            self.obs.clear()
            self.act.clear()
            self.rew.clear()
            self.val.clear()
            self.logp.clear()
            self.adv = np.empty(0)
            self.ret = np.empty(0)

        def store(self, observation, action, value, logp, reward=0) -> None:
            self.obs.append(observation)
            self.act.append(action)
            self.rew.append(reward)
            self.val.append(value)
            self.logp.append(logp)
            self.curr_ptr += 1

        def store_past_reward(self, reward) -> None:
            if self.curr_ptr > 0:
                self.rew[-1] = reward

        def on_end_trajectory(self, gamma, lam, last_reward) -> None:
            self.store_past_reward(last_reward)
            trajectory_slice = slice(self.start_ptr, self.curr_ptr)

            # Compute advantages and rewards-to-go
            rewards = np.array(self.rew[trajectory_slice], dtype=np.float32)
            values = np.array(self.val[trajectory_slice], dtype=np.float32)
            self.total_rewards = np.sum(rewards)
            values = np.append(values, 0) # This acts as our mask to avoid taking into account the last value

            def discounted_sum(array, discount):
                # Based on https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation
                y = lfilter([1], [1, -discount], x=array[::-1])
                return y[::-1]

            deltas = rewards + (gamma * values[1:]) - values[:-1]
            advantages = discounted_sum(deltas, gamma * lam)
            expected_returns = discounted_sum(rewards, gamma)

            self.adv = np.append(self.adv, advantages)
            self.ret = np.append(self.ret, expected_returns)

            # Advance
            self.start_ptr = self.curr_ptr

        def on_end_simulation(self) -> dict:
            """
            Reset and normalize advantages

            Returns collected data
            """

            # Normalize advantages
            adv_mean = np.mean(self.adv)
            adv_std = np.std(self.adv)
            self.adv = (self.adv - adv_mean) / adv_std

            self.dict['adv'] = self.adv
            self.dict['ret'] = self.ret

            # Create list of samples
            sample_dict = {}
            for k, v in self.dict.items():
                if k == 'obs':
                    v = torch.stack(v)
                sample_dict[k] = torch.as_tensor(v, dtype=torch.float32, device=DEVICE)

            # Reset
            self._reset()
            return sample_dict

    class Actor(nn.Module):
        def __init__(self, observation_size, wait_action):
            super(EnergyActorCritic.Actor, self).__init__()
            self.wait_action = wait_action
            self.model = nn.Sequential(
                nn.Linear(observation_size[1], 128),
                nn.GELU(),
                nn.Linear(128, 64, device=DEVICE),
                nn.GELU(),
                nn.Linear(64, 32, device=DEVICE),
                nn.GELU(),
                nn.Linear(32, 16, device=DEVICE),
                nn.GELU(),
                nn.Linear(16, 8, device=DEVICE),
                nn.GELU(),
                nn.Linear(8, 1, device=DEVICE),
                nn.GELU()
            ).to(DEVICE)

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

            approx_kl = (logp_old - logp).mean().item()
            return loss, approx_kl

    class Critic(nn.Module):
        def __init__(self, observation_size):
            super(EnergyActorCritic.Critic, self).__init__()
            self.model0 = nn.Sequential(
                nn.Linear(observation_size[1], 4),
                nn.GELU(),
                nn.Linear(4, 2),
                nn.GELU(),
                nn.Linear(2, 1),
                nn.GELU()
            ).to(DEVICE)
            self.model1 = nn.Sequential(
                nn.Linear(observation_size[0], 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 8),
                nn.GELU(),
                nn.Linear(8, 1),
                nn.GELU()
            ).to(DEVICE)

        def forward(self, observation: torch.Tensor) -> torch.Tensor:

            # First phase output: dim = (num_batches, all job-node combinations (actions_size))
            out = self.model0(observation)
            out = torch.squeeze(out)

            # Second phase:
            return self.model1(out)

        def loss(self, minibatch: dict):
            obs = minibatch['obs']
            ret = minibatch['ret']
            if ret.shape[0] != 1:
                ret = ret.unsqueeze(1)
            loss = torch.nn.MSELoss()
            return loss(self.forward(obs), ret).mean()

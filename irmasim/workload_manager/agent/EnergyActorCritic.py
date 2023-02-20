import numpy as np
import random
from typing import Tuple, Any

from irmasim.Options import Options
from scipy.signal import lfilter
import os.path as path
import torch
import torch.nn as nn
from torch.distributions import Categorical

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: can a trajectory end before trajectory_length steps?
# TODO: where require_grad=True?


class EnergyActorCritic:
    def __init__(self, actions_size: tuple, observation_size: tuple, load_agent: bool = True):

        random.seed(3)  # TODO in options maybe
        options = Options().get()
        self.agent_options = Options().get()['workload_manager']['agent']

        self.actor = EnergyActorCritic.Actor(observation_size[1])
        self.critic = EnergyActorCritic.Critic(actions_size[0], observation_size[1])  # TODO figure this out

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.agent_options['lr_pi']))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(self.agent_options['lr_v']))

        self.lam = self.agent_options['lambda']
        self.gamma = self.agent_options['gamma']

        if self.agent_options['phase'] == 'train':
            self.train_pi_iters = self.agent_options['train_pi_iters']
            self.train_v_iters = self.agent_options['train_v_iters']

        self.buffer = EnergyActorCritic.PPOBuffer(options['trajectory_length'],
                                                  options['nbtrajectories'])

        if 'input_model' in self.agent_options \
                and path.isfile(self.agent_options['input_model']) \
                and load_agent:
            in_model = self.agent_options['input_model']
            print(f'Reading model to {in_model}')
            checkpoint = torch.load(in_model)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['pi'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['v'])
            # TODO: load an existing model, train or eval? Also something with the optimizers

    def decide(self, observation: torch.Tensor) -> tuple:
        with torch.no_grad():  # We don't want to compute gradients for a forward pass
            action, logp_action, _ = self.actor.forward(observation.to(DEVICE))
            value = self.critic.forward(observation.to(DEVICE))
        return action, value, logp_action

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])

    @property
    def total_rewards(self):
        return self.buffer.total_rewards

    def on_end_trajectory(self):
        self.buffer.on_end_trajectory(self.gamma, self.lam)

    def training_step(self, num_minibatches=32, val_factor=0.85, h_factor=0.08):  # TODO in options
        """
        Training step, executed after each simulation

        Based on the code of the guy who did his master's thesis on PPO,
        and on ActionActorCritic
        """
        num_samples = self.buffer.capacity
        assert num_samples % num_minibatches == 0
        minibatch_size = num_samples // num_minibatches

        # Normalize advantages, reset buffer and collect data
        training_data = self.buffer.on_end_simulation()
        losses = np.zeros((self.train_pi_iters, 3), dtype=np.float32)

        def next_minibatch():
            for i in range(0, num_samples, minibatch_size):
                m = {}
                for k in ['obs', 'act', 'rew', 'val', 'logp', 'adv', 'ret']:
                    m[k] = training_data[k][i:i+minibatch_size]

                # Think of this as a static variable in C/C++
                next_minibatch.counter += minibatch_size
                yield m
        next_minibatch.counter = 0  # Can't be initialized inside the function

        for epoch in range(self.train_pi_iters):  # TODO entropy
            for minibatch in next_minibatch():
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                loss_pi = self.actor.loss(minibatch)
                loss_v = self.critic.loss(minibatch)
                _, _, entropy = self.actor.forward(minibatch['obs'])
                final_loss = loss_pi + val_factor * loss_v - h_factor * entropy
                final_loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                losses[epoch][0] = loss_pi.item()
                losses[epoch][1] = loss_v.item()
                losses[epoch][2] = final_loss.item()

        return np.mean(losses, axis=0)

    class PPOBuffer:
        def __init__(self, trajectory_length, nbtrajectories):
            self.trajectory_length = trajectory_length
            self.nbtrajectories = nbtrajectories
            self.obs = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.act = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.rew = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.val = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.logps = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.adv = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.ret = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.dict = {'obs': self.obs, 'act': self.act, 'rew': self.rew, 'val': self.val,
                         'logps': self.logps, 'adv': self.adv, 'ret': self.ret}

            self.traj_idx = 0
            self.timestep = 0
            self.total_rewards = 0

        @property
        def capacity(self):
            return self.trajectory_length * self.nbtrajectories

        def store(self, observation, action, reward, value, logp):
            self.obs[self.traj_idx][self.timestep] = observation
            self.act[self.traj_idx][self.timestep] = action
            self.rew[self.traj_idx][self.timestep] = reward
            self.val[self.traj_idx][self.timestep] = value
            self.logps[self.traj_idx][self.timestep] = logp
            self.timestep += 1

        def on_end_trajectory(self, gamma, lam) -> None:
            assert self.timestep == self.trajectory_length

            # Compute advantages and rewards-to-go
            rewards = self.rew[self.traj_idx]
            self.total_rewards = np.sum(rewards)
            values = self.val[self.traj_idx]
            values.append(0)  # This acts as our mask to avoid taking into account the last value

            def discounted_sum(array, discount):
                # Based on https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation
                y = lfilter([1], [1, -discount], x=array[::-1])
                return y[::-1]  # Reversed?

            # TODO understand this completely
            deltas = rewards + (gamma * values[1:]) - values[:-1]
            advantages = discounted_sum(deltas, gamma * lam)
            expected_returns = discounted_sum(rewards, gamma)

            self.adv[self.traj_idx] = advantages
            self.ret[self.traj_idx] = expected_returns

            # Advance to next row
            self.traj_idx += 1
            self.timestep = 0

            """
            for t in reversed(range(trajectory_length)):  # TODO mask
                mask = 0 if t == trajectory_length else 1
                delta = rewards[t] + (self.gamma * values[t + 1] * mask) - values[t]
                advantages[t] = delta + (self.gamma * self.lam * advantages[t + 1] * mask)
                #delta = rewards[i] + self.gamma * values[i + 1] - values[i]
                #gae = delta + self.gamma * self.lam * gae
                #returns.insert(0, gae + values[i])

            adv = np.array(returns) - values[:-1]
            """

        def on_end_simulation(self) -> dict:
            """
            Reset and normalize advantages

            Returns collected data
            """
            assert self.traj_idx == self.nbtrajectories

            # Reset
            self.traj_idx = 0
            self.timestep = 0

            # Normalize advantages
            adv_mean = np.mean(self.adv)
            adv_std = np.std(self.adv)
            self.adv = (self.adv - adv_mean) / adv_std

            # Create shuffled list of samples
            # Flatten outer dimension and shuffle each array the same way
            sample_dict = {}
            seed = random.random()
            for k, v in self.dict.items():
                arr = v.reshape(-1, v.shape[-1])
                np.random.seed(seed)
                np.random.shuffle(arr)  # TODO quite expensive, right?
                sample_dict[k] = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)

            return sample_dict

    class Actor(nn.Module):
        def __init__(self, observation_size):
            super(EnergyActorCritic.Actor, self).__init__()
            self.input = nn.Linear(observation_size, 32, device=DEVICE)
            self.hidden_0 = nn.Linear(32, 16, device=DEVICE)
            self.hidden_1 = nn.Linear(16, 8, device=DEVICE)
            self.output = nn.Linear(8, 1, device=DEVICE)

        def forward(self, observation: torch.Tensor, act=None) -> Tuple[Any, Any, Any]:
            # TODO: investigate how the mask works and so on
            mask = torch.where(observation.sum(dim=-1) != 0.0, 1.0, 0.0)
            print(f'actor mask: {mask}')
            out_0 = nn.functional.leaky_relu(self.input(observation))
            out_1 = nn.functional.leaky_relu(self.hidden_0(out_0))
            out_2 = nn.functional.leaky_relu(self.hidden_1(out_1))
            out_3 = torch.squeeze(self.output(out_2), dim=-1)
            out = out_3 + (mask - 1) * 1e6
            # TODO: check how the distribution looks, might need to normalize

            pi = Categorical(logits=out)
            print(f'Distribution: {pi.logits}')
            action = pi.sample() if act is None else act
            print(f'Action: {action}')
            return action, pi.log_prob(action), pi.entropy()

        def loss(self, minibatch: dict, epsilon=0.2):  # TODO in options?
            obs = minibatch['obs']
            act = minibatch['act']
            adv = minibatch['adv']
            logp_old = minibatch['logp']

            _, logp, _ = self.forward(obs, act)
            ratio = torch.exp(logp - logp_old)
            clipped = torch.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
            loss = torch.minimum(ratio * adv, clipped)
            return -torch.mean(loss)

    class Critic(nn.Module):
        def __init__(self, actions_size, observation_size):
            super(EnergyActorCritic.Critic, self).__init__()

            # TODO wth is this architecture
            self.input = nn.Linear(observation_size, 32, device=DEVICE)
            self.hidden_0_0 = nn.Linear(32, 16, device=DEVICE)
            self.hidden_0_1 = nn.Linear(16, 8, device=DEVICE)
            self.hidden_0_2 = nn.Linear(8, 1, device=DEVICE)

            self.hidden_1_0 = nn.Linear(actions_size, 64, device=DEVICE)
            self.hidden_1_1 = nn.Linear(64, 32, device=DEVICE)
            self.hidden_1_2 = nn.Linear(32, 8, device=DEVICE)
            self.output = nn.Linear(8, 1, device=DEVICE)

        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            print(f'Critic observation: {observation}, shape: {observation.shape}')
            out_0_0 = nn.functional.leaky_relu(self.input(observation))
            out_0_1 = nn.functional.leaky_relu(self.hidden_0_0(out_0_0))
            out_0_2 = nn.functional.leaky_relu(self.hidden_0_1(out_0_1))
            out_0_3 = nn.functional.leaky_relu(self.hidden_0_2(out_0_2))
            out_1 = torch.squeeze(out_0_3)
            print(f'Critic out_0_0 shape: {out_0_0.shape}')
            # print(out_0_0)
            print(f'Critic out_0_1 shape: {out_0_1.shape}')
            # print(out_0_1)
            print(f'Critic out_0_2 shape: {out_0_2.shape}')
            # print(out_0_2)
            print(f'Critic out_0_3 shape: {out_0_3.shape}')
            # print(out_0_3)
            print(f'Critic out_1 shape: {out_1.shape}')

            out_1_1 = nn.functional.leaky_relu(self.hidden_1_0(out_1))
            out_1_2 = nn.functional.leaky_relu(self.hidden_1_1(out_1_1))
            out_1_3 = nn.functional.leaky_relu(self.hidden_1_2(out_1_2))

            return self.output(out_1_3)

        def loss(self, minibatch: dict):
            obs = minibatch['obs']
            ret = minibatch['ret']
            loss = torch.nn.MSELoss()
            return loss(self.forward(obs), ret)

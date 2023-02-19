import numpy as np
from typing import Tuple, Any

from irmasim.Options import Options
from scipy.signal import lfilter
import os.path as path
import torch
import torch.nn as nn
from torch.distributions import Categorical

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: can a trajectory end before trajectory_length steps?

class EnergyActorCritic:
    def __init__(self, actions_size: int, observation_size: int, gamma=0.99, lam=0.97, load_agent: bool = True):
        self.agent_options = Options().get()['workload_manager']['agent']

        self.actor = EnergyActorCritic.Actor(observation_size)
        self.critic = EnergyActorCritic.Critic(actions_size, observation_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.agent_options['lr_pi']))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(self.agent_options['lr_v']))

        self.lam = self.agent_options['lambda']
        self.gamma = self.agent_options['gamma']

        if self.agent_options['phase'] == 'train':
            self.train_pi_iters = self.agent_options['train_pi_iters']
            self.train_v_iters = self.agent_options['train_v_iters']

        self.buffer = EnergyActorCritic.PPOBuffer(actions_size, observation_size,
                                                  self.agent_options['trajectory_length'],
                                                  self.agent_options['nbtrajectories'])

        if 'input_model' in self.agent_options \
                and path.isfile(self.agent_options['input_model']) \
                and load_agent:
            in_model = self.agent_options['input_model']
            print(f'Reading model to {in_model}')
            checkpoint = torch.load(in_model)
            self.load_state_dict(checkpoint['model_state_dict'])
            # TODO: load an existing model, train or eval?

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])

    def on_end_trajectory(self):
        # Compute advantages and rewards-to-go
        rewards = self.buffer.get_current_batch('rew')
        values = self.buffer.get_current_batch('val')
        values.append(0)  # This acts as our mask to avoid taking into account the last value

        def discounted_sum(array, discount):
            # Based on https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation
            y = lfilter([1], [1, -discount], x=array[::-1])
            return y[::-1]  # Reversed?

        # TODO understand this completely
        deltas = rewards + (self.gamma * values[1:]) - values[:-1]
        advantages = discounted_sum(deltas, self.gamma * self.lam)
        expected_returns = discounted_sum(rewards, self.gamma)

        self.buffer.batch_store('adv', advantages)
        self.buffer.batch_store('ret', expected_returns)

        # Advance to next row in buffer
        self.buffer.advance_trajectory()

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

    def training_step(self):
        """
        Training step, executed after each simulation
        """

        # Normalize advantages, reset buffer and collect data
        training_data = self.buffer.simulation_end()

        # Train actor and critic using collected data (sampled in minibatches)

        for i in range(self.train_pi_iters):
            loss = 0  # TODO
            self.actor_optimizer.zero_grad()


        # Train critic

    def decide(self, observation: torch.Tensor) -> tuple:
        with torch.no_grad():  # We don't want to compute gradients for a forward pass
            action, logp_action = self.actor.forward(observation.to(DEVICE))
            value = self.critic.forward(observation.to(DEVICE))
        return action, value, logp_action

    class PPOBuffer:
        # TODO: polish this to make it more modular and maintainable
        def __init__(self, actions_size, observation_size, trajectory_length, nbtrajectories):
            self.trajectory_length = trajectory_length
            self.nbtrajectories = nbtrajectories

            self.obs = np.zeros((nbtrajectories, trajectory_length, observation_size), dtype=np.float32)
            self.act = np.zeros((nbtrajectories, trajectory_length, actions_size), dtype=np.float32)

            self.rew = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.val = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.logps = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)

            self.adv = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)
            self.ret = np.zeros((nbtrajectories, trajectory_length), dtype=np.float32)

            self.dict = {'obs': self.obs, 'act': self.act, 'rew': self.rew, 'val': self.val,
                         'logps': self.logps, 'adv': self.adv, 'ret': self.ret}

            self.traj_idx = 0
            self.timestep = 0

        def store(self, observation, action, reward, value, logp):
            self.obs[self.traj_idx][self.timestep] = observation
            self.act[self.traj_idx][self.timestep] = action
            self.rew[self.traj_idx][self.timestep] = reward
            self.val[self.traj_idx][self.timestep] = value
            self.logps[self.traj_idx][self.timestep] = logp
            self.timestep += 1

        def batch_store(self, key, batch_values):
            """
            Stores at current trajectory index
            """
            self.dict[key][self.traj_idx] = batch_values

        def advance_trajectory(self):
            self.traj_idx += 1
            self.timestep = 0

        def get_current_batch(self, key: str) -> np.array:
            # Only finished batches can be accesed
            assert self.timestep == self.trajectory_length
            return self.dict[key][self.traj_idx]

        def simulation_end(self) -> np.array:
            """
            Reset and normalize advantages

            Returns collected data as an array of tuples
            """
            self.traj_idx = 0
            self.timestep = 0
            adv_mean = np.mean(self.adv)
            adv_std = np.std(self.adv)
            self.adv = (self.adv - adv_mean) / adv_std

            for
            return {k: torch.as_tensor(np.squeeze(v, dim=0), device=DEVICE) for k, v in self.dict}


    class Actor(nn.Module):
        def __init__(self, observation_size):
            super(EnergyActorCritic.Actor, self).__init__()
            self.input = nn.Linear(observation_size, 32, device=DEVICE)
            self.hidden_0 = nn.Linear(32, 16, device=DEVICE)
            self.hidden_1 = nn.Linear(16, 8, device=DEVICE)
            self.output = nn.Linear(8, 1, device=DEVICE)

        def forward(self, observation: torch.Tensor) -> Tuple[Any, Any]:
            # TODO: investigate how the mask works and so on
            mask = torch.where(observation.sum(dim=-1) != 0.0, 1.0, 0.0)
            out_0 = nn.functional.leaky_relu(self.input(observation))
            out_1 = nn.functional.leaky_relu(self.hidden_0(out_0))
            out_2 = nn.functional.leaky_relu(self.hidden_1(out_1))
            out_3 = torch.squeeze(self.output(out_2), dim=-1)
            out = out_3 + (mask - 1) * 1e6
            # TODO: check how the distribution looks, might need to normalize

            pi = Categorical(logits=out)
            action = pi.sample()
            return action, pi.log_prob(action)

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
            out_0_0 = nn.functional.leaky_relu(self.input(observation))
            out_0_1 = nn.functional.leaky_relu(self.hidden_0_0(out_0_0))
            out_0_2 = nn.functional.leaky_relu(self.hidden_0_1(out_0_1))
            out_0_3 = nn.functional.leaky_relu(self.hidden_0_2(out_0_2))
            out_1 = torch.squeeze(out_0_3)

            out_1_1 = nn.functional.leaky_relu(self.hidden_1_0(out_1))
            out_1_2 = nn.functional.leaky_relu(self.hidden_1_1(out_1_1))
            out_1_3 = nn.functional.leaky_relu(self.hidden_1_2(out_1_2))

            return self.output(out_1_3)

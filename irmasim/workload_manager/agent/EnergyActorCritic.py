import numpy as np
import os.path as path
import torch
import torch.nn as nn
from irmasim.Options import Options
from scipy.signal import lfilter
from torch.distributions import Categorical
from torch.optim.swa_utils import AveragedModel, SWALR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BUFFER_SIZE_MPLIER = 5


def init_weights(module):
    """
    Used to initialize the weights of the actor and critic networks.
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.bias is not None:
            module.bias.data.zero_()

class EnergyActorCritic:
    """
    Implements the decision-making module (agent) of the Energy scheduler using Proximal Policy Optimization.

    - Actor and critic are self-normalizing neural networks.
    - Adam optimization is used.
    - Stochastic Weight Averaging may be enabled.
    - The scheduler may choose to wait even when enough resources are available to schedule a job,
      if this option is enabled.
    """
    def __init__(self, observation_size: tuple, load_agent: bool = True):

        # Initialize actor, critic and buffer
        self.agent_options = Options().get()['workload_manager']['agent']
        lam = self.agent_options['lambda']
        gamma = self.agent_options['gamma']
        wait_action = Options().get()['workload_manager']['wait_action']
        buffer_size = Options().get()['nbtrajectories'] * Options().get()['trajectory_length']
        if wait_action:
            buffer_size *= BUFFER_SIZE_MPLIER
        self.buffer = PPOBuffer(observation_size, buffer_size, lam, gamma)
        self.actor = EnergyActorCritic.Actor(observation_size, wait_action)
        self.critic = EnergyActorCritic.Critic(observation_size)

        # Initialize actor and critic optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(self.agent_options['lr_pi']))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(self.agent_options['lr_v']))


        # Configure Stochastic Weight Averaging
        self.swa_on = False
        self.actor_swa_model = None
        self.SWA_START_PC = 0.75
        self.SWA_ANNEALING_PC = 0.15
        if 'swa' in self.agent_options and self.agent_options['swa']:
            self.swa_on = self.agent_options['swa']
            lr_pi = float(self.agent_options['lr_pi'])
            iters = int(self.agent_options['train_iters'])
            self.swa_start_epoch = int(self.SWA_START_PC * iters)
            self.actor_swa_scheduler = SWALR(self.actor_optimizer,
                                             anneal_strategy="linear",
                                             anneal_epochs=int(self.SWA_ANNEALING_PC * iters),
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

    def state_dict(self) -> dict:
        return {
            'actor': self.actor_swa_model.state_dict() if self.actor_swa_model else self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.actor.load_state_dict(state_dict['actor'], strict=False)
        self.critic.load_state_dict(state_dict['critic'])

    def store(self, observation, action, value, logp, reward=0) -> None:
        self.buffer.store(observation, action, value, logp, reward)

    def decide(self, observation: torch.Tensor) -> tuple:
        # Gradient information is not necessary when generating training data
        with torch.no_grad():
            action, logp_action, _ = self.actor.forward(observation.to(DEVICE))
            value = self.critic.forward(observation.to(DEVICE))
        return action.item(), value.item(), logp_action.item()

    @property
    def total_rewards(self) -> float:
        # For the last trajectory only
        return self.buffer.total_rewards

    def reward_last_action(self, reward) -> None:
        self.buffer.store_past_reward(reward)

    def on_end_trajectory(self, last_reward) -> None:
        """
        Computes advantages and expected returns in the buffer.
        """
        self.buffer.on_end_trajectory(last_reward)

    def training_step(self) -> np.array:
        """
        Updates the parameters of the actor and critic networks based on the collected training data.

        Returns: the mean actor and critic losses during the current training simulation.
        """
        num_samples = self.buffer.num_samples
        minibatch_size = self.agent_options['minibatch_size']
        train_iters = self.agent_options['train_iters']
        h_factor = self.agent_options['h_factor']
        epsilon = self.agent_options['clipping_factor']

        # Normalize advantages, reset buffer and collect training data
        training_data = self.buffer.collect()
        losses = np.zeros((train_iters, 2), dtype=np.float32)

        def next_minibatch():
            """
            Fetches the next minibatch.
            """
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
        """
        Actor network.
        Given an observation of the environment, predicts an action in the form of a job-node assignment.
        """
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

        def forward(self, observation: torch.Tensor, act=None) -> tuple:
            mask = torch.where(observation.sum(dim=-1) != 0.0, 1.0, 0.0)
            if self.wait_action:
                mask[-1] = 1.0
            out = self.model(observation)
            out = torch.squeeze(out, dim=-1)
            out = out + (mask - 1.0) * 1e6
            pi = Categorical(logits=out)
            action = pi.sample() if act is None else act
            return action, pi.log_prob(action), torch.mean(pi.entropy())

        def loss(self, minibatch: dict, epsilon) -> float:
            """
            Implements the PPO Clip loss function.
            """
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
        """
        Critic network.
        Given an observation of the environment, predicts the value of the current state.
        """
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

            # First phase: reduce from OBSERVATION_SIZE to ACTIONS_SIZE
            out_0_0 = nn.functional.selu(self.input(observation))
            out_0_1 = nn.functional.selu(self.hidden_0_0(out_0_0))
            out_0_2 = nn.functional.selu(self.hidden_0_1(out_0_1))
            out_0_3 = nn.functional.selu(self.hidden_0_2(out_0_2))
            out_0_4 = nn.functional.selu(self.hidden_0_3(out_0_3))
            out_1 = torch.squeeze(out_0_4)

            # Second phase: reduce from ACTIONS_SIZE to a single value
            out_1_1 = nn.functional.selu(self.hidden_1_0(out_1))
            out_1_2 = nn.functional.selu(self.hidden_1_1(out_1_1))
            out_1_3 = nn.functional.selu(self.hidden_1_2(out_1_2))
            out_1_4 = nn.functional.selu(self.hidden_1_3(out_1_3))
            out_1_5 = nn.functional.selu(self.hidden_1_4(out_1_4))
            out = self.output(out_1_5)
            return out

        def loss(self, minibatch: dict):
            """
            Simple MSE loss.
            """
            obs = minibatch['obs']
            ret = minibatch['ret']
            if ret.shape[0] != 1:
                ret = ret.unsqueeze(1)
            loss = torch.nn.MSELoss()
            return loss(self.forward(obs), ret).mean()



class PPOBuffer:
    """
    Buffer used to collect the training data required to implement Proximal Policy Optimization.
    """
    def __init__(self, observation_size, buffer_size, lam, gamma):
        self.start_ptr = 0
        self.curr_ptr = 0
        self.max_size = buffer_size
        self.lam = lam
        self.gamma = gamma
        self.total_rewards = 0  # For the last trajectory only

        # Initialize arrays
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

    def on_end_trajectory(self, last_reward) -> None:
        """
        Stores the last reward of the trajectory and computes the corresponding
        advantages and expected returns.
        """
        self.store_past_reward(last_reward)

        # Compute advantages and rewards-to-go
        trajectory_slice = slice(self.start_ptr, self.curr_ptr)
        rewards = self.rew[trajectory_slice]
        values = self.val[trajectory_slice]
        self.total_rewards = np.sum(rewards)

        # Acts as a mask to avoid taking into account the last value when computing the advantages
        values = np.append(values, 0)

        def discounted_sum(array, discount):
            y = lfilter([1], [1, -discount], x=array[::-1])
            return y[::-1]

        deltas = rewards + (self.gamma * values[1:]) - values[:-1]
        advantages = discounted_sum(deltas, self.gamma * self.lam)
        expected_returns = discounted_sum(rewards, self.gamma)

        self.adv[trajectory_slice] = advantages
        self.ret[trajectory_slice] = expected_returns

        # Advance
        self.start_ptr = self.curr_ptr

    def collect(self) -> dict:
        """
        Normalizes the advantages and returns the collected samples.

        Returns: a dictionary of tensors containing all the collected samples.
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
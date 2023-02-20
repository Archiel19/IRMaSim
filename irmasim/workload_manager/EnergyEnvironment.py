import importlib
import gym
import logging

import torch
from gym import spaces
from gym.core import ActType, ObsType, RenderFrame
import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional, Union, List
from irmasim.workload_manager import EnergyWM

if TYPE_CHECKING:
    from irmasim.workload_manager.EnergyWM import EnergyWM
from irmasim.Simulator import Simulator
from irmasim.Options import Options


class EnergyEnvironment(gym.Env):
    """
    Based on ActionEnvironment.py and Environment.py

    Observation: node and job attributes

    Job attributes:
    - Number of requested processors (cores?)
    - Requested time
    - Wait time

    Node attributes:
    - Static power
    - Dynamic power
    - Clock rate
    - Can be scheduled
    """

    def __init__(self, workload_manager: 'EnergyWM', simulator: Simulator):
        super(EnergyEnvironment, self).__init__()
        self.simulator = simulator
        self.options = Options().get()
        self.workload_manager = workload_manager
        env_options = self.options["workload_manager"]["environment"]
        self.NUM_JOBS = env_options["num_jobs"]
        self.NUM_NODES = env_options["num_nodes"]
        self.OBS_FEATURES = env_options["obs_features"]

        mod = importlib.import_module("irmasim.platform.models." + self.options["platform_model_name"] + ".Node")
        klass = getattr(mod, 'Node')
        self.resources = self.simulator.get_resources(klass)

        # Set reward function
        reward_dict = {
            'energy_consumption': self._energy_consumption_reward,
            'edp': self._edp_reward
        }
        objective = env_options['objective']
        if objective not in reward_dict:
            all_objectives = ", ".join(reward_dict.keys())
            raise Exception(f"Unknown objective {objective}. Must be one of: {all_objectives}.")
        self.reward = reward_dict[objective]

        # Action space (match a job with a node)
        self.action_space = spaces.Discrete(self.NUM_JOBS * self.NUM_NODES)

        # Observation space TODO
        # For each node: dyn power, static power, num cores, can be scheduled
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(self.NUM_JOBS * self.NUM_NODES, self.OBS_FEATURES),
                                                dtype=np.float32)
        # TODO: initial observation?

    def _get_obs(self) -> torch.Tensor:
        return torch.zeros([1280, 7])

    def _get_info(self):
        # TODO
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), self._get_info()

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Apply action
        print(action)
        job_idx, node_idx = self._get_action_pair(action)
        print(job_idx)
        logging.getLogger("irmasim").debug(
            f"{self.simulator.simulation_time} performing action Job({job_idx})-Node({node_idx}) ({action})")
        # TODO: do the resources have a mutable state? Is it safe to have this sort of duplication with the WM?

        print('Pending jobs [Environment]:')
        for j in self.workload_manager.pending_jobs:
            print(j)
        job, node = self.workload_manager.pending_jobs.pop(job_idx), self.resources[node_idx]
        mod = importlib.import_module('irmasim.platform.models.' + self.options['platform_model_name'] + '.Core')
        core_klass = getattr(mod, 'Core')  # TODO maybe put this somewhere else
        free_cores = [core for core in node.enumerate_resources(core_klass) if core.task is None]

        print(f'free cores: {len(free_cores)}, job ntasks: {job.ntasks}')
        assert len(free_cores) >= job.ntasks
        for task in job.tasks:
            task.allocate(free_cores.pop(0).full_id())
        self.simulator.schedule(job.tasks)
        self.workload_manager.running_jobs.append(job)  # TODO: modularity

        # Retrieve return values
        observation = self._get_obs()
        done = False  # This can only be known from the workload manager
        info = self._get_info()
        return observation, self.reward, done, False, info

    def _energy_consumption_reward(self) -> float:
        delta_time = self.simulator.simulation_time - self.workload_manager.last_time
        return -self.simulator.platform.get_joules(delta_time)

    def _edp_reward(self) -> float:
        makespan = self.workload_manager.last_time - self.simulator.simulation_time
        return self._energy_consumption_reward() * makespan

    def _get_action_pair(self, action: torch.Tensor) -> tuple:
        # Job-node pairs can be viewed as a matrix
        action = action.item()
        job = action // self.NUM_NODES
        node = action % self.NUM_NODES
        print(f'action: {action}, job: {job}, node: {node}')
        return job, node

    @property
    def actions_size(self) -> tuple:
        return self.action_space.n,

    @property
    def observation_size(self) -> tuple:
        return self.observation_space.shape



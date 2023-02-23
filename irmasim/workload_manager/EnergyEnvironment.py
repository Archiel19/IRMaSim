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

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

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

        mod = importlib.import_module('irmasim.platform.models.' + self.options['platform_model_name'] + '.Core')
        self.core_klass = getattr(mod, 'Core')

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

        # Observation space
        # For each node: dyn power, static power, num cores, can be scheduled
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(self.NUM_JOBS * self.NUM_NODES, self.OBS_FEATURES),
                                                dtype=np.float32)

    def get_obs(self) -> torch.Tensor:
        # TODO make my version
        mod = importlib.import_module("irmasim.platform.models." + self.options["platform_model_name"] + ".Core")
        klass = getattr(mod, 'Core')
        observation = []
        for job in self.workload_manager.pending_jobs[:self.NUM_JOBS]:
            wait_time = self.simulator.simulation_time - job.submit_time
            req_time = job.req_time
            req_core = job.ntasks
            job_obs = [wait_time, req_time, req_core]

            for node in self.resources:
                core_list = node.enumerate_resources(klass)
                available_core_list = []
                clock_rate_sum = 0
                for core in core_list:
                    clock_rate_sum += core.clock_rate
                    if core.task is None:
                        available_core_list.append(core)
                avg_clock_rate = clock_rate_sum / len(core_list)
                if req_core <= len(available_core_list):
                    observation.append(job_obs + [len(core_list), len(available_core_list), avg_clock_rate,
                                                  int(req_core <= len(available_core_list))])
                else:
                    observation.append([0] * 7)

        num_fill_jobs = self.actions_size[0] - len(observation)
        # No pad on top, pad 'num_fill_jobs' to the bottom, no pad left nor right
        return torch.Tensor(np.pad(observation, [(0, num_fill_jobs), (0, 0)]))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # TODO?
        return self.get_obs(), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def apply_action(self, action: ActType) -> None:
        # Apply action
        job_idx, node_idx = self._get_action_pair(action)
        logging.getLogger("irmasim").debug(
            f"{self.simulator.simulation_time} performing action Job({job_idx})-Node({node_idx}) ({action})")

        job, node = self.workload_manager.pending_jobs.pop(job_idx), self.resources[node_idx]
        free_cores = [core for core in node.enumerate_resources(self.core_klass) if core.task is None]

        # print(f'free cores: {len(free_cores)}, job ntasks: {job.ntasks}')
        assert len(free_cores) >= job.ntasks
        for task in job.tasks:
            task.allocate(free_cores.pop(0).full_id())
        self.simulator.schedule(job.tasks)
        self.workload_manager.running_jobs.append(job)  # TODO: modularity

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
        return job, node

    @property
    def actions_size(self) -> tuple:
        return self.action_space.n,

    @property
    def observation_size(self) -> tuple:
        return self.observation_space.shape



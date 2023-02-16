import importlib
import gym
from gym import spaces
import numpy as np
from typing import TYPE_CHECKING

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

    def __init__(self, workload_manager: EnergyWM, simulator: Simulator):
        super(EnergyEnvironment, self).__init__()
        self.workload_manager = workload_manager
        self.simulator = simulator
        self.options = Options().get()

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

    def _energy_consumption_reward(self) -> float:
        delta_time = self.simulator.simulation_time - self.workload_manager.last_time
        return -self.simulator.platform.get_joules(delta_time)

    def _edp_reward(self) -> float:
        makespan = self.workload_manager.last_time - self.simulator.simulation_time
        return self._energy_consumption_reward() * makespan

    @property
    def actions_size(self) -> tuple:
        return self.action_space.shape

    @property
    def observation_size(self) -> tuple:
        return self.observation_space.shape

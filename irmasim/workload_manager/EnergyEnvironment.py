import importlib
import gym
import logging

import torch
from gym import spaces
import numpy as np
from irmasim.Simulator import Simulator
from irmasim.Options import Options


class EnergyEnvironment:
    """
    Acts as a proxy of the simulated instance of the cluster.

    Observation space: [NUM_JOBS * NUM_NODES, NUM_FEATURES]
    Action space: [NUM_JOBS * NUM_NODES]

    Features considered for each job-node pair:

    Job features:
    - Submit time
    - Number of requested cores
    - Requested time
    - Wait time

    Node features:
    - Availability = Free processors / Total processors
    - Static power
    - Dynamic power
    - Clock rate
    """

    def __init__(self, simulator: Simulator):
        super(EnergyEnvironment, self).__init__()
        self.simulator = simulator
        self.options = Options().get()
        env_options = self.options["workload_manager"]["environment"]
        mod = importlib.import_module("irmasim.platform.models." + self.options["platform_model_name"] + ".Node")
        klass = getattr(mod, 'Node')
        self.resources = self.simulator.get_resources(klass)
        self.NUM_NODES = len(self.resources)
        self.NUM_JOBS = env_options["num_jobs"]
        self.OBS_FEATURES = 9
        self.pending_jobs = []

        # Static node attributes
        self.static_power = dict.fromkeys(self.resources)
        self.idle_power = dict.fromkeys(self.resources)
        self.clock_rate = dict.fromkeys(self.resources)
        self.assigned_jobs = dict.fromkeys(self.resources)
        self.job_node_pairings = dict()
        for node in self.resources:
            self.static_power[node] = sum([core.static_power for core in node.cores()])
            self.idle_power[node] = sum([core.min_power * core.static_power for core in node.cores()])
            self.clock_rate[node] = sum([core.clock_rate for core in node.cores()]) / len(node.cores())
            self.assigned_jobs[node] = set()
        self.f_min = min(self.clock_rate.values())

        # Action space (match a job with a node)
        wait_action = 1 if self.options["workload_manager"]["wait_action"] else 0
        self.action_space = spaces.Discrete(self.NUM_JOBS * self.NUM_NODES + wait_action) # + 1 for 'wait'

        # Observation space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(self.actions_size[0], self.OBS_FEATURES),
                                                dtype=np.float32)

        # Extra action
        self.WAIT_ACTION = self.actions_size[0] -1 if wait_action else -1

        # Set reward function
        self.last_total_energy = 0.0
        self.last_used_energy = 0.0
        self.last_time = 0.0
        reward_dict = {
            'energy_consumption': self._energy_consumption_reward,
            'edp': self._edp_reward
        }
        objective = env_options['objective']
        if objective not in reward_dict:
            all_objectives = ", ".join(reward_dict.keys())
            raise Exception(f"Unknown objective {objective}. Must be one of: {all_objectives}.")
        self.reward = reward_dict[objective]

    def get_obs(self) -> torch.Tensor:
        """
        Job attributes: (stay the same)
        - Number of requested processors (cores?)
        - Requested time
        - Submit time
        - Wait time

        Node attributes:
        - Free processors / total processors
        - Static power
        - Dynamic power
        - Clock rate
        """
        observation = np.zeros(self.observation_size, dtype=np.float64)
        for j, job in enumerate(self.pending_jobs[:self.NUM_JOBS]):
            wait_time = self.simulator.simulation_time - job.submit_time
            job_obs = [wait_time, job.req_time, job.submit_time, job.ntasks]
            for n, node in enumerate(self.resources):
                available_cores = node.count_idle_cores()
                obs = []
                if job.ntasks <= available_cores:
                    energy_est = self.energy_estimate(job, node)
                    if energy_est <= job.max_energy:
                        availability = available_cores / len(node.cores())
                        avg_clock_rate = self.clock_rate[node]
                        static_power = self.static_power[node] if available_cores < len(node.cores()) \
                            else self.idle_power[node]
                        dynamic_power = sum([core.dynamic_power for core in node.cores() if core.task])
                        obs = job_obs + [availability, avg_clock_rate, static_power, dynamic_power, energy_est]
                for i, val in enumerate(obs):
                    observation[j * self.NUM_NODES + n][i] = val

        # Normalize
        norm_sum = observation.sum()
        if norm_sum:
            observation = np.divide(observation, norm_sum)
        return torch.Tensor(observation)

    def reset(self):
        self.last_total_energy = 0.0
        self.last_used_energy = 0.0
        self.last_time = 0.0
        self.pending_jobs = []

    def add_jobs(self, jobs):
        self.pending_jobs += jobs

    def finish_jobs(self, finished_jobs):
        for job in finished_jobs:
            node = self.job_node_pairings[job.id]
            self.assigned_jobs[node].remove(job.id)
            del self.job_node_pairings[job.id]

    def can_schedule(self):
        for job in self.pending_jobs[:self.NUM_JOBS]:
            # Don't change condition order
            suitable_nodes = [node for node in self.resources
                              if node.count_idle_cores() >= job.ntasks
                              and self.energy_estimate(job, node) <= job.max_energy]
            if suitable_nodes:
                return True
        return False

    def energy_estimate(self, job, node):
        f = self.clock_rate[node]
        t_est = (self.f_min / f) * job.req_time

        num_jobs = len(self.assigned_jobs[node]) + 1  # + 1 for the target job
        s_power = self.static_power[node] / num_jobs

        free_cores = [core for core in node.cores() if not core.task]
        d_power = sum([core.dynamic_power for core in free_cores[:job.ntasks]])

        return t_est * (s_power + d_power)

    def apply_action(self, action: int) -> None:
        if action == self.WAIT_ACTION:
            logging.getLogger("irmasim").debug(f"{self.simulator.simulation_time} waiting...")
            self.simulator.set_alarm(1.0)
            return
        self.simulator.unset_alarm()
        job_idx, node_idx = self._get_action_pair(action)
        if job_idx >= len(self.pending_jobs):
            print(f'Job: {job_idx}, node: {node_idx}, len pending: {len(self.pending_jobs)}')
        logging.getLogger("irmasim").debug(
            f"{self.simulator.simulation_time} performing action Job({job_idx})-Node({node_idx}) ({action})")

        job = self.pending_jobs.pop(job_idx)
        node = self.resources[node_idx]
        free_cores = [core for core in node.cores() if core.task is None]

        assert len(free_cores) >= job.ntasks
        core = 0
        for task in job.tasks:
            task.allocate(free_cores[core].full_id())
            core += 1
        self.assigned_jobs[node].add(job.id)
        self.job_node_pairings[job.id] = node
        self.simulator.schedule(job.tasks)

    def _update_reward_vars(self):
        self.last_total_energy = self.simulator.total_energy
        self.last_used_energy = self.simulator.used_energy
        self.last_time = self.simulator.simulation_time

    def _energy_consumption_reward(self, last_reward=False) -> float:
        total_energy_incr = self.simulator.total_energy - self.last_total_energy
        self._update_reward_vars()
        return -total_energy_incr

    def _edp_reward(self, last_reward=False) -> float:
        makespan = self.simulator.simulation_time - self.last_time
        return self._energy_consumption_reward(last_reward) * makespan

    def _get_action_pair(self, action: int) -> tuple:
        # Job-node pairs can be viewed as a matrix
        job = action // self.NUM_NODES
        node = action % self.NUM_NODES
        return job, node

    @property
    def actions_size(self) -> tuple:
        return self.action_space.n,

    @property
    def observation_size(self) -> tuple:
        return self.observation_space.shape

import importlib
import logging
import torch
from irmasim.workload_manager.WorkloadManager import WorkloadManager
from irmasim.workload_manager.EnergyEnvironment import EnergyEnvironment
from irmasim.workload_manager.agent.EnergyActorCritic import EnergyActorCritic
from irmasim.Options import Options
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from irmasim.Simulator import Simulator


class EnergyWM(WorkloadManager):
    """
    Actor-critic architecture focused on minimizing energy expenditure etc etc
    """
    def __init__(self, simulator: 'Simulator'):
        super(EnergyWM, self).__init__(simulator)
        if simulator.platform.config["model"] != "modelV1":
            raise Exception("EnergyWM workload manager needs a modelV1 platform")
        self.options = Options().get()
        mod = importlib.import_module("irmasim.platform.models." + self.options["platform_model_name"] + ".Node")
        klass = getattr(mod, 'Node')

        self.resources = self.simulator.get_resources(klass)
        self.pending_jobs = []
        self.running_jobs = []

        # DRL-related attributes
        self.environment = EnergyEnvironment(self, simulator)
        self.agent = EnergyActorCritic(self.environment.actions_size,
                                       self.environment.observation_size)

    def on_job_submission(self, jobs: list):
        self.pending_jobs += jobs

    def on_job_completion(self, jobs: list):
        for job in jobs:
            logging.getLogger("irmasim").debug(
                f"{self.simulator.simulation_time} {job.name} finished")
            self.running_jobs.remove(job)

    def on_end_step(self):
        if self._can_schedule():
            self.agent.reward_last_action(self.environment.reward())
            observation = self.environment.get_obs()
            action, value, logp = self.agent.decide(observation)
            self.environment.apply_action(action, self.pending_jobs, self.running_jobs)

            # Rewards can only be computed after the simulator has applied the action,
            # so there's a separate function in the agent to reward the last taken action
            self.agent.buffer.store(observation, action, value, logp)

    def _can_schedule(self):
        for job in self.pending_jobs[:self.environment.NUM_JOBS]:
            if job.ntasks <= max([node.count_idle_cores() for node in self.resources]):
                return True
        return False

    def on_end_trajectory(self):
        logging.getLogger('irmasim').debug(f'{self.simulator.simulation_time} - Ending trajectory')
        self.agent.on_end_trajectory(self.environment.reward())
        self.environment.reset()

    def on_end_simulation(self):
        phase = self.options['workload_manager']['agent']['phase']
        out_dir = self.options['output_dir']
        if phase == 'train':
            # Compute losses
            losses = self.agent.training_step()
            with open(f'{out_dir}/losses.log', 'a+') as out_f:
                out_f.write(f'{losses[0]}, {losses[1]}, {losses[2]}\n')
            if 'output_model' in self.options['workload_manager']['agent']:
                out_model = self.options['workload_manager']['agent']['output_model']
                print(f"Writing model to {out_model}")
                torch.save({
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': {'pi': self.agent.actor_optimizer.state_dict(),
                                             'v': self.agent.critic_optimizer.state_dict()}
                }, out_model)
        with open(f'{out_dir}/rewards.log', 'a+') as out_f:
            out_f.write(f'{self.agent.total_rewards}\n')

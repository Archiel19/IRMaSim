import importlib
import logging
import torch
from irmasim.workload_manager.WorkloadManager import WorkloadManager
from irmasim.workload_manager.EnergyEnvironment import EnergyEnvironment
from irmasim.workload_manager.agent.EnergyActorCritic import EnergyActorCritic
from irmasim.Options import Options
from irmasim.Task import Task
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
        self.last_time = 0

        # DRL-related attributes
        self.environment = EnergyEnvironment(self, simulator)
        # TODO: the sizes
        self.agent = EnergyActorCritic(self.environment.actions_size[0],
                                       self.environment.observation_size[0])
        self.observation = self.environment.reset()[0]

    def on_job_submission(self, jobs: list):
        self.pending_jobs += jobs

    def on_job_completion(self, jobs: list):
        for job in jobs:
            logging.getLogger("irmasim").debug(
                f"{self.simulator.simulation_time} {job.name} finished")
            self.running_jobs.remove(job)

    # TODO The following two have been taken from NodeWM, make sure to understand what this function does wrt the sim
    def schedule_next_job(self):
        if self.pending_jobs and max([resource[1] for resource in self.resources]) > 0:
            next_job = self.pending_jobs.pop(0)
            for task in next_job.tasks:
                self.allocate(task)
            self.simulator.schedule(next_job.tasks)
            self.running_jobs.append(next_job)
            return True
        else:
            return False

    def allocate(self, task: Task):
        resource = 0
        while self.resources[resource][1] == 0:
            resource += 1
        self.resources[resource][1] -= 1
        task.allocate(self.resources[resource][0])

    def on_end_step(self):
        if self._can_schedule():
            self.last_time = self.simulator.simulation_time
            action, value, logp = self.agent.decide(self.observation)
            self.observation, reward, _, _, _ = self.environment.step(action)
            self.agent.buffer.store(self.observation, action, reward, value, logp)

    def _can_schedule(self):  # TODO: figure out how this works
        for job in self.pending_jobs[:self.environment.NUM_JOBS]:
            if job.ntasks <= max(self.resources, key=lambda node: node.count_idle_cores()):
                return True
        return False

    def on_end_trajectory(self):
        logging.getLogger('irmasim').debug(f'{self.simulator.simulation_time} - Ending trajectory')
        self.agent.on_end_trajectory()
        # TODO when is the environment reset?

    def on_end_simulation(self):
        phase = self.options['workload_manager']['agent']['phase']
        out_dir = self.options['output_dir']
        if phase == 'train':
            # Compute losses
            losses = self.agent.training_step()  # TODO this, check if f'' needed
            with open(f'{out_dir}', 'a+') as out_f:
                out_f.write(f'{losses[0]}, {losses[1]}\n')
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

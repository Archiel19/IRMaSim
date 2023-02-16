import importlib
import logging
from irmasim.workload_manager.WorkloadManager import WorkloadManager
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

        # TODO DRL-specific attributes
        # self.environment = Environment(self, simulator)
        # self.agent, self.optimizer = self.create_agent()

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
        self.last_time = self.simulator.simulation_time

    def on_end_trajectory(self):
        pass

    def on_end_simulation(self):
        pass

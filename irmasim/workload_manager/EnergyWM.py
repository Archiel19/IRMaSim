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
    Energy scheduler.

    Implements an intelligent actor-critic reinforcement learning agent devoted to minimizing energy consumption
    or EDP making use of the power specifications of the resources and an enegy estimate.

    This class is mainly responsible for coordinating calls between EnergyActorCritic and EnergyEnvironment.
    """
    def __init__(self, simulator: 'Simulator'):
        super(EnergyWM, self).__init__(simulator)
        if simulator.platform.config["model"] != "modelV1":
            raise Exception("EnergyWM workload manager needs a modelV1 platform")
        self.options = Options().get()

        # RL-related attributes
        self.trajectory_start = True
        self.environment = EnergyEnvironment(simulator)
        self.agent = EnergyActorCritic(self.environment.observation_size)

    def on_job_submission(self, jobs: list):
        self.environment.add_jobs(jobs)

    def on_job_completion(self, jobs: list):
        self.environment.finish_jobs(jobs)
        for job in jobs:
            logging.getLogger("irmasim").debug(
                f"{self.simulator.simulation_time} {job.name} finished")

    def on_end_step(self):
        if self.environment.can_schedule():
            if self.trajectory_start:
                # Set flag so that the next rewards will be stored
                self.trajectory_start = False
            else:
                # Rewards can only be computed after the simulator has applied the previous action
                self.agent.reward_last_action(self.environment.reward())

            observation = self.environment.get_obs()
            action, value, logp = self.agent.decide(observation)
            self.environment.apply_action(action)
            self.agent.store(observation, action, value, logp)

    def on_alarm(self):
        self.on_end_step()

    def on_end_trajectory(self):
        logging.getLogger('irmasim').debug(f'{self.simulator.simulation_time} - Ending trajectory')
        self.agent.on_end_trajectory(self.environment.reward(last_reward=True))
        self.environment.reset()
        self.trajectory_start = True

    def on_end_simulation(self):
        phase = self.options['workload_manager']['agent']['phase']
        out_dir = self.options['output_dir']

        if phase == 'train':
            # Train actor and critic for the specified number of epochs/iterations
            losses = self.agent.training_step()
            with open(f'{out_dir}/losses.log', 'a+') as out_f:
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
        with open(f'{out_dir}/edp.log', 'a+') as out_f:
            out_f.write(f'{self.simulator.total_energy * self.simulator.simulation_time}\n')
        with open(f'{out_dir}/energy.log', 'a+') as out_f:
            out_f.write(f'{self.simulator.total_energy}\n')

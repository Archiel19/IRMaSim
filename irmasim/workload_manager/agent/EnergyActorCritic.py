from irmasim.workload_manager.agent import Agent
from irmasim.Options import Options
import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EnergyActorCritic:
    def __init__(self):
        self.options = Options().get()
        self.actor = EnergyActorCritic.Actor(self)
        self.critic = EnergyActorCritic.Critic(self)

    class Actor(Agent.PolicyLearningAgent):
        def __init__(self, parent):
            super(EnergyActorCritic.Actor, self).__init__(parent.options)

        def forward_policy(self, observation: np.ndarray) -> torch.Tensor:
            pass

    class Critic(Agent.ValueLearningAgent):
        def __init__(self, parent):
            super(EnergyActorCritic.Critic, self).__init__(parent.options)

        def forward_value(self, observation: np.ndarray) -> torch.Tensor:
            pass

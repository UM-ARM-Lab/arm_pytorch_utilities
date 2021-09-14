import numpy as np
import abc
from arm_pytorch_utilities import serialization


class Controller(serialization.Serializable):
    """
    Controller that gives a command for a given observation (public API is ndarrays)
    Internally may keep state represented as ndarrays or tensors
    """

    def __init__(self, compare_to_goal=np.subtract):
        """
        :param compare_to_goal: function (state, goal) -> diff batched difference
        """
        self.goal = None
        self.compare_to_goal = compare_to_goal

    def state_dict(self) -> dict:
        return {'goal': self.goal}

    def load_state_dict(self, state: dict) -> bool:
        self.goal = state['goal']
        return True

    def reset(self):
        """Clear any controller state to be reused in another trial"""

    def get_goal(self):
        return self.goal

    def set_goal(self, goal):
        self.goal = goal

    @abc.abstractmethod
    def command(self, obs, info=None):
        """Given current observation and misc info, command an action"""

    def get_rollouts(self, obs):
        """Return what the predicted states for the selected action sequence is applied on obs"""
        return None

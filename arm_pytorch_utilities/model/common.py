import abc
import logging
import os

import torch
from arm_pytorch_utilities.optim import Lookahead

logger = logging.getLogger(__name__)


class LearnableParameterizedModel:
    def __init__(self, save_root_dir, name='', lookahead=False):
        self.save_root_dir = save_root_dir
        self.optimizer = None
        self.step = 0
        self.name = name
        self.optimizer = torch.optim.Adam(self.parameters())
        if lookahead:
            self.optimizer = Lookahead(self.optimizer)

    @abc.abstractmethod
    def parameters(self):
        """
        :return: Iterable holding this transform's parameters
        """

    @abc.abstractmethod
    def _model_state_dict(self):
        """
        :return: State dictionary of the model to save
        """

    @abc.abstractmethod
    def _load_model_state_dict(self, saved_state_dict):
        """
        Load saved state dictionary
        :param saved_state_dict: what _model_state_dict returns
        :return:
        """

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, last=False):
        state = {
            'step': self.step,
            'state_dict': self._model_state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        base_dir = os.path.join(self.save_root_dir, 'checkpoints')
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        full_name = os.path.join(base_dir, '{}.{}.tar'.format(self.name, self.step))
        torch.save(state, full_name)
        if last:
            logger.info("saved checkpoint %s", full_name)

    def load(self, filename):
        if not os.path.isfile(filename):
            return False
        checkpoint = torch.load(filename)
        self.step = checkpoint['step']
        self._load_model_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return True

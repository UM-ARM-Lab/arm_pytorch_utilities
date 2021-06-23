import logging
import os
import itertools

import torch
from arm_pytorch_utilities import array_utils
from arm_pytorch_utilities.optim import Lookahead

logger = logging.getLogger(__name__)


# TODO make subclass of nn.Module?
class LearnableParameterizedModel:
    def __init__(self, save_root_dir, name='', lookahead=False):
        self.save_root_dir = save_root_dir
        self.step = 0
        self.name = name
        params = self.parameters()
        self.optimizer = torch.optim.Adam(params if len(params) else [torch.zeros(1)])
        if lookahead:
            self.optimizer = Lookahead(self.optimizer)

    def modules(self):
        """PyTorch compatible modules that make it convenient to define the other methods"""
        return {}

    def parameters(self):
        """
        :return: Iterable holding this transform's parameters
        """
        return list(itertools.chain.from_iterable(module.parameters() for module in self.modules().values()))

    @property
    def device(self):
        return self.parameters()[0].device

    @property
    def dtype(self):
        return self.parameters()[0].dtype

    def _model_state_dict(self):
        """
        :return: State dictionary of the model to save
        """
        return {key: module.state_dict() for key, module in self.modules().items()}

    def _load_model_state_dict(self, saved_state_dict):
        """
        Load saved state dictionary
        :param saved_state_dict: what _model_state_dict returns
        :return:
        """
        modules = self.modules()
        try:
            for key, module in modules.items():
                module.load_state_dict(saved_state_dict[key])
        except KeyError as e:
            logger.error("Abort loading due to %s", e)
            return False
        return True

    def eval(self):
        for param in self.parameters():
            param.requires_grad = False

    def train(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_last_checkpoint(self, sort_by_time=True):
        """
        Get the last checkpoint for a model matching this one's name (with the highest number of training steps)
        :return: either '' or the filename of the last checkpoint
        """
        base_dir = os.path.join(self.save_root_dir, 'checkpoints')
        # look for files with this base name, what follows should be step.tar
        # (use . after name to prevent matching prefix)
        checkpoints = [filename for filename in os.listdir(base_dir) if filename.startswith(self.name + '.')]
        if not checkpoints:
            return ''
        if sort_by_time:
            checkpoints = [os.path.join(base_dir, filename) for filename in checkpoints]
            checkpoints.sort(key=os.path.getmtime)
            return checkpoints[-1]
        else:
            # order by name
            array_utils.sort_nicely(checkpoints)
            return os.path.join(base_dir, checkpoints[-1])

    def save(self, last=False):
        """
        Save the model to a checkpoint
        :param last: whether it's the last save of the training and to log where we saved to
        :return:
        """
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
        """
        Load the model from the filename for a checkpoint
        :param filename:
        :return: whether we successfully loaded the checkpoint or not
        """
        if not os.path.isfile(filename):
            return False
        try:
            checkpoint = torch.load(filename)
        except RuntimeError as e:
            logger.warning(e)
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        self.step = checkpoint['step']
        if not self._load_model_state_dict(checkpoint['state_dict']):
            return False
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("loaded checkpoint at step %d for %s", self.step, self.name)
        return True

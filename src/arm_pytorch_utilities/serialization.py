import torch
import os.path
import logging

logger = logging.getLogger(__name__)


class Serializable():
    """
    Stateful objects that can be saved and loaded
    """

    def save(self, filename):
        """Save any state collected during the run (excludes data given at construction time for simplicity)"""
        state = self.state_dict()
        base_dir = os.path.dirname(filename)
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        torch.save(state, filename)
        logger.info("saved checkpoint to %s", filename)

    def load(self, filename) -> bool:
        """Load state previously saved, returning loading success"""
        if not os.path.isfile(filename):
            return False
        try:
            checkpoint = torch.load(filename)
        except RuntimeError as e:
            logger.warning(e)
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        try:
            if not self.load_state_dict(checkpoint):
                logger.info("failed to load checkpoint from %s", filename)
                logger.info(checkpoint)
                return False
        except KeyError as e:
            logger.error(e)
            logger.info(checkpoint)
            return False
        logger.info("loaded checkpoint from %s", filename)
        return True

    def state_dict(self) -> dict:
        """State collected during the run"""
        return {}

    def load_state_dict(self, state: dict) -> bool:
        return True

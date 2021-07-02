"""Framework for compatible simulations under the /simulation directory"""
import random
import time
import logging
import os
import scipy.io
import pybullet as p
import pybullet_data
import abc

from arm_pytorch_utilities import rand

logger = logging.getLogger(__name__)


class Mode:
    DIRECT = 0
    GUI = 1


class ReturnMeaning:
    SUCCESS = 0
    ERROR = 1
    REJECTED = 2


class _DefaultConfig:
    DATA_DIR = './data'


class Simulation(abc.ABC):
    def __init__(self, save_dir='raw', mode=Mode.GUI, log_video=False, plot=False, save=False,
                 num_frames=300, sim_step_s=1. / 240., config=_DefaultConfig):
        # simulation meta
        self.save_dir = os.path.join(config.DATA_DIR, save_dir)
        self.mode = mode
        self.log_video = log_video

        # simulation config
        self.num_frames = num_frames
        self.sim_step_s = sim_step_s

        # actions to do
        self.plot = plot
        self.save = save

        # per run state variables
        self.randseed = None

    def run(self, randseed=None, run_name=None):
        if randseed is None:
            rand.seed(int(time.time()))
            randseed = random.randint(0, 1000000)
        logger.debug('random seed: %d', randseed)
        self.randseed = randseed
        rand.seed(randseed)

        ret = self._configure_physics_engine()
        if ret is not ReturnMeaning.SUCCESS:
            return ret

        ret = self._setup_experiment()
        if ret is not ReturnMeaning.SUCCESS:
            return ret

        ret = self._init_data()
        if ret is not ReturnMeaning.SUCCESS:
            return ret

        ret = self._run_experiment()
        if ret is not ReturnMeaning.SUCCESS:
            return ret

        # plot data
        if self.plot:
            self._plot_data()

        # save experiment
        if self.save:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            run_name = run_name if run_name is not None else randseed
            save_to = os.path.join(self.save_dir, '{}.mat'.format(run_name))
            # export in matlab/numpy compatible format
            scipy.io.savemat(save_to, mdict=self._export_data_dict())
            logger.info("Finished saving to {}".format(save_to))

        return ReturnMeaning.SUCCESS

    @abc.abstractmethod
    def _configure_physics_engine(self):
        return ReturnMeaning.SUCCESS

    def _setup_experiment(self):
        return ReturnMeaning.SUCCESS

    def _init_data(self):
        return ReturnMeaning.SUCCESS

    @abc.abstractmethod
    def _run_experiment(self):
        return ReturnMeaning.SUCCESS

    def _plot_data(self):
        pass

    def _export_data_dict(self):
        return {}


class PyBulletSim(Simulation):
    def __init__(self, realtime_simulation=False, **kwargs):
        super(PyBulletSim, self).__init__(**kwargs)
        self.physics_client = None
        self.realtime = realtime_simulation

    def _configure_physics_engine(self):
        mode_dict = {Mode.GUI: p.GUI, Mode.DIRECT: p.DIRECT}

        # if the mode we gave is in the dict then use it, otherwise use the given mode value as is
        mode = mode_dict.get(self.mode) or self.mode

        self.physics_client = p.connect(mode)  # p.GUI for GUI or p.DIRECT for non-graphical version

        if self.log_video:
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "{}.mp4".format(self.randseed))

        # use data provided by PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        if self.realtime:
            p.setRealTimeSimulation(True)
        else:
            p.setTimeStep(self.sim_step_s)

        return ReturnMeaning.SUCCESS

    def run(self, randseed=None, run_name=None):
        # make sure to always disconnect after running
        ret = super(PyBulletSim, self).run(randseed, run_name)
        p.disconnect(self.physics_client)
        return ret

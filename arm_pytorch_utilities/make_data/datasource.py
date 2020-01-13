import abc
from typing import Type

import torch
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities import str_utils


class DataSource:
    """
    Source of data, agnostic to its creation method; public API for data
    """

    def __init__(self, N=None, config=load_data.DataConfig(), use_gpu_if_available=False):
        self.N = N
        self.config = config

        # GPU speedup
        if use_gpu_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._train = None
        self._val = None

    def training_set(self):
        return self._train

    def validation_set(self):
        return self._val

    @abc.abstractmethod
    def make_data(self):
        """Create data that'll be available from training_set and validation_set"""

    def data_id(self):
        """String identification for this data"""
        return "N_{}".format(str_utils.f2s(self.N))


class FileDataSource(DataSource):
    def __init__(self, loader: Type[load_data.DataLoader], data_dir, preprocessor: preprocess.Preprocess = None,
                 validation_ratio=0.2,
                 **kwargs):
        """
        :param loader: data loader specializing to what's stored in each file
        :param data_dir: data directory or list of directories
        :param predict_difference: whether the output should be the state differences or states
        :param preprocessor: data preprocessor, such as StandardizeVariance
        :param validation_ratio: amount of data set aside for validation
        :param kwargs:
        """

        super().__init__(**kwargs)

        self.loader = loader
        self.preprocessor = preprocessor
        self._data_dir = data_dir
        self._validation_ratio = validation_ratio
        self.make_data()

    def make_data(self):
        full_set = load_data.LoaderXUYDataset(loader=self.loader, dirs=self._data_dir, max_num=self.N,
                                              config=self.config)
        train_set, validation_set = load_data.split_train_validation(full_set,
                                                                     validation_ratio=self._validation_ratio)

        self.N = len(train_set)

        if self.preprocessor:
            self.preprocessor.fit(train_set)
            self.preprocessor.update_data_config(self.config)
            # apply on training and validation set
            train_set = self.preprocessor.transform(train_set)
            validation_set = self.preprocessor.transform(validation_set)

        self._train = load_data.get_all_data_from_dataset(train_set)
        self._val = load_data.get_all_data_from_dataset(validation_set)

    def data_id(self):
        """String identification for this data"""
        return "{}_N_{}_{}".format(self._data_dir, str_utils.f2s(self.N), self.config)

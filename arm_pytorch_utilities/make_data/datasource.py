import abc
import copy
from typing import Type

import torch
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities import preprocess
from arm_pytorch_utilities import str_utils


class DataSource:
    """
    Source of data, agnostic to its creation method; public API for data
    """

    def __init__(self, max_N=None, config=load_data.DataConfig(), preprocessor: preprocess.Transformer = None,
                 device="cpu"):
        """
        :param N: number of data points for this data source; set to None for unknown/loaded
        :param config: configuration of data of this source
        :param preprocessor: data transformer/preprocessor, such as StandardizeVariance
        :param device:
        """
        self.max_N = max_N
        self.N = max_N
        self.config = config
        self._original_config = None

        self.d = torch.device(device)

        self._train = None
        self._val = None
        # data before preprocessing; set only if we have a preprocessor
        self._original_train = None
        self._original_val = None

        self.preprocessor = None  # implementation should use preprocessor in make_data
        self.update_preprocessor(preprocessor)

    def training_set(self, original=False):
        return self._original_train if original and self._original_train else self._train

    def validation_set(self, original=False):
        return self._original_val if original and self._original_val else self._val

    def original_validation_set(self):
        return self.validation_set(original=True)

    @abc.abstractmethod
    def make_data(self):
        """Create data that'll be available from training_set and validation_set"""

    def data_id(self):
        """String identification for this data"""
        return "N_{}".format(str_utils.f2s(self.N))

    def original_config(self):
        return self.config if self._original_config is None else self._original_config

    def current_config(self):
        return self.config

    def update_preprocessor(self, preprocessor):
        """Change the preprocessor,
        which involves remaking the data and potentially changing the meaning/config of the data.
        If you need the loaded untransformed config (for example for doing control in that space), then you should
        not pass the preprocessor in the constructor, but instead call this function with it afterwards.

        :param preprocessor:
        :return: original data configuration (before *any* transformation, even if this is not the first transform)
        """
        # the first time we're updating the preprocessor and the original data has been loaded
        if self._original_config is None and self.config.nx is not None:
            self._original_config = copy.deepcopy(self.config)

        self.preprocessor = preprocess.DatasetPreprocessor(preprocessor) if \
            isinstance(preprocessor, preprocess.Transformer) else preprocessor
        self.make_data()
        return self.original_config()


class FileDataSource(DataSource):
    def __init__(self, loader: load_data.DataLoader, data_dir, validation_ratio=0.2, **kwargs):
        """
        :param loader: data loader specializing to what's stored in each file
        :param data_dir: data directory or list of directories
        :param predict_difference: whether the output should be the state differences or states
        :param validation_ratio: amount of data set aside for validation
        :param kwargs:
        """
        self.loader = loader
        self._data_dir = data_dir
        self._validation_ratio = validation_ratio
        super().__init__(**kwargs)

    def make_data(self):
        full_set = load_data.LoaderXUYDataset(loader=self.loader, dirs=self._data_dir, max_num=self.max_N,
                                              config=self.config, device=self.d)
        train_set, validation_set = load_data.split_train_validation(full_set,
                                                                     validation_ratio=self._validation_ratio)

        self.N = len(train_set)

        if self.preprocessor:
            self.preprocessor.fit(train_set)
            self.preprocessor.update_data_config(self.config)
            # save old data (if it's for the first time we're using a preprocessor)
            if self._original_val is None:
                self._original_train = train_set[:]
                self._original_val = validation_set[:]
            # apply on training and validation set
            train_set = self.preprocessor.transform(train_set)
            validation_set = self.preprocessor.transform(validation_set)

        self._train = train_set[:]
        self._val = validation_set[:]

    def restrict_training_set_to_slice(self, restricted_slice):
        if self._train is not None:
            self._train = tuple(v[restricted_slice] for v in self._train)

    def data_id(self):
        """String identification for this data"""
        return "{}_N_{}_{}".format(self._data_dir, str_utils.f2s(self.N), self.config)

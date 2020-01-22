import abc
import os
from typing import Type

import numpy as np
import scipy.io
import torch
import torch.utils.data


class DataConfig:
    """
    Data class holding configuration information about a dataset
    """

    def __init__(self, sort_data=True, predict_difference=True, predict_all_dims=True, force_affine=False,
                 expanded_input=False, y_in_x_space=True):
        """

        :param sort_data: Whether the experiments (data between files) are sorted (by random seed id)
        :param predict_difference: Whether the prediction should be the state difference or the next state
        :param predict_all_dims: Whether the prediction should include all state dimensions (some datasets do this regardless)
        :param force_affine: Whether a column of 1s should be added to XU to make the dataset effectively affine
        :param expanded_input: Whether the input has extra dimensions (packed in control dimension)
        :param y_in_x_space: Whether the output is in the same space as state; if not then a lot of assumptions will not hold
        """
        self.sort_data = sort_data
        self.predict_difference = predict_difference
        self.predict_all_dims = predict_all_dims
        self.force_affine = force_affine
        self.expanded_input = expanded_input
        self.y_in_x_space = y_in_x_space
        # unknown quantities until we encounter data (optional)
        self.nx = None
        self.nu = None
        self.ny = None
        # sometimes the full input is larger than xu (such as with expanded input)
        self.n_input = None

    def load_data_info(self, x, u=None, y=None, full_input=None):
        self.nx = x.shape[1]
        if u is not None:
            self.nu = u.shape[1]
        if y is not None:
            self.ny = y.shape[1]
        if full_input is not None:
            self.n_input = full_input.shape[1]
        if self.expanded_input and full_input is None:
            raise RuntimeError("Need to load full input with expanded input")

    def input_dim(self):
        if not self.nx:
            raise RuntimeError("Need to load data info first before asking for input dim")
        if self.n_input:
            return self.n_input
        # else assume we're inputting xu
        ni = self.nx
        if self.nu:
            ni += self.nu
        return ni

    def options(self):
        return self.sort_data, self.predict_difference, self.predict_all_dims, self.force_affine, \
               self.expanded_input, self.y_in_x_space

    def __str__(self):
        return "i{}_o{}_s{}_pd{}_pa{}_a{}_e{}_y{}".format(self.input_dim(), self.ny,
                                                          *(int(config) for config in self.options()))

    def __repr__(self):
        return "DataConfig(sort_data={}, predict_difference={}, predict_all_dims={}, force_affine={}, " \
               "expanded_input={}, y_in_x_space={})".format(*self.options())


class DataLoader(abc.ABC):
    """
    Driver for loading a dataset from file.
    Each dataset should subclass DataLoader and specialize process file raw data to saved content.
    """

    def __init__(self, file_cfg=None, dir_to_load=None, config=DataConfig()):
        if file_cfg is None or dir_to_load is None:
            raise RuntimeError("Incomplete specification of DataLoader")
        self.dir = dir_to_load
        self.data = None
        self.config = config
        self.file_cfg = file_cfg

    @abc.abstractmethod
    def _process_file_raw_data(self, d):
        """
        Turn a file's dictionary content into a data sequence, each element of which has the same number of rows
        :param d: file's dictionary content
        :return: tuple of data sequence
        """

    def load_file(self, full_filename):
        raw_data = scipy.io.loadmat(full_filename)
        file_data = self._process_file_raw_data(raw_data)
        if self.data is None:
            self.data = list(file_data)
        else:
            for i in range(len(self.data)):
                self.data[i] = np.row_stack((self.data[i], file_data[i]))
        return self.data

    def load(self):
        full_dir = os.path.join(self.file_cfg.DATA_DIR, self.dir)

        if os.path.isfile(full_dir):
            self.load_file(full_dir)
        else:
            files = os.listdir(full_dir)
            # consistent with the way MATLAB loads files
            if self.config.sort_data:
                files = sorted(files)

            for file in files:
                full_filename = '{}/{}'.format(full_dir, file)
                if os.path.isdir(full_filename):
                    continue
                self.load_file(full_filename)
        return self.data


def make_affine(X):
    N = X.shape[0]
    return torch.cat((X, torch.ones((N, 1), dtype=X.dtype)), dim=1)


class RandomNumberDataset(torch.utils.data.Dataset):
    def __init__(self, produce_output, num=1000, low=-1, high=1, input_dim=1):
        r = high - low
        self.x = torch.rand((num, input_dim)) * r / 2 + (low + high) / 2
        self.y = produce_output(self.x)
        super(RandomNumberDataset, self).__init__()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_input(self):
        return self.x

    def get_output(self):
        return self.y


class SimpleXUYDataset(torch.utils.data.Dataset):
    def __init__(self, XU, Y):
        self.XU = XU
        self.Y = Y
        super(SimpleXUYDataset, self).__init__()

    def __len__(self):
        return len(self.XU)

    def __getitem__(self, idx):
        return self.XU[idx], self.Y[idx]


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, *sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences[0])

    def __getitem__(self, idx):
        return tuple(sequence[idx] if sequence is not None else [] for sequence in self.sequences)


class IndexedDataset(SimpleDataset):
    """Same as before, but with last element as the index of the data point for using in Dataloaders"""

    def __getitem__(self, idx):
        return tuple(sequence[idx] if sequence is not None else [] for sequence in self.sequences) + (idx,)


class PartialViewDataset(torch.utils.data.Dataset):
    """Get a slice of a full dataset (for example to split training and validation set)
    taken from https://discuss.pytorch.org/t/best-way-training-data-in-pytorch/6855/2"""

    def __init__(self, full_data, offset, length):
        self.data = full_data
        self.offset = offset
        self.length = length
        assert len(full_data) >= offset + length, Exception("View of dataset goes outside full dataset")
        super(PartialViewDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise StopIteration()
        return self.data[idx + self.offset]


class LoaderXUYDataset(torch.utils.data.Dataset):
    def __init__(self, loader: Type[DataLoader], dirs=('raw',), filter_on_labels=None, max_num=None,
                 config=DataConfig(), device="cpu"):
        if type(dirs) is str:
            dirs = [dirs]
        self.XU = None
        self.Y = None
        self.labels = None
        for dir in dirs:
            dl = loader(dir_to_load=dir, config=config)
            XU, Y, labels = dl.load()
            if self.XU is None:
                self.XU = XU
                self.Y = Y
                self.labels = labels
            else:
                self.XU = np.row_stack((self.XU, XU))
                self.Y = np.row_stack((self.Y, Y))
                self.labels = np.row_stack((self.labels, labels))
        self._convert_types(device)
        if filter_on_labels:
            self.XU, self.Y, self.labels = filter_on_labels(self.XU, self.Y, self.labels)

        if config.force_affine:
            self.XU = make_affine(self.XU)

        if max_num is not None:
            self.XU = self.XU[:max_num]
            self.Y = self.Y[:max_num]
            self.labels = self.labels[:max_num]

        super().__init__()

    def _convert_types(self, device):
        self.XU = torch.from_numpy(self.XU).double().to(device=device)
        self.Y = torch.from_numpy(self.Y).double().to(device=device)
        self.labels = torch.from_numpy(self.labels).byte().to(device=device)

    def __len__(self):
        return self.XU.shape[0]

    def __getitem__(self, idx):
        return self.XU[idx], self.Y[idx], self.labels[idx]


def split_train_validation(dataset, validation_ratio=0.1):
    # consider giving a shuffle (with np.random.shuffle()) option to permute the data before viewing
    offset = int(len(dataset) * (1 - validation_ratio))
    return PartialViewDataset(dataset, 0, offset), PartialViewDataset(dataset, offset, len(dataset) - offset)


def merge_data_in_dir(config, dir, out_filename, sort=True):
    full_dir = os.path.join(config.DATA_DIR, dir)

    files = os.listdir(full_dir)
    if sort:
        files = sorted(files)

    data = None
    for file in files:
        full_filename = '{}/{}'.format(full_dir, file)
        raw_data = scipy.io.loadmat(full_filename)
        if data is None:
            data = raw_data
        else:
            for key in raw_data.keys():
                data[key] = np.row_stack((data[key], raw_data[key]))
    merged_filename = '{}/{}.mat'.format(config.DATA_DIR, out_filename)
    scipy.io.savemat(merged_filename, data)


def get_all_data_from_dataset(dataset):
    x0, y0, ls = dataset[0]
    XU = x0.new_zeros((len(dataset), x0.shape[0]))
    Y = x0.new_zeros((len(dataset), y0.shape[0]))
    labels = torch.zeros(len(dataset), dtype=ls.dtype, device=x0.device)
    for i, data in enumerate(dataset, 0):
        xu, y, ls = data
        XU[i] = xu
        Y[i] = y
        labels[i] = ls
    return XU, Y, labels

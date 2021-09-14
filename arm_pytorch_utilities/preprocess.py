import abc
import copy
import logging
import typing
from typing import Iterable

import torch
from arm_pytorch_utilities import load_data
from arm_pytorch_utilities import tensor_utils

logger = logging.getLogger(__name__)


class Transformer(abc.ABC):
    """Like sklearn's Transformers
    Different from pytorch's transforms as those are typically applied as you retrieve the data (usually images)
    """

    def __init__(self):
        # allow only fit once; for refit, use a different instance of the preprocessor
        # usually refits are mistakes
        self.fitted = False

    def fit(self, XU, Y, labels=None):
        if self.fitted:
            logger.warning("Ignoring attempt to refit preprocessor")
            return
        self._fit_impl(XU, Y, labels)
        self.fitted = True

    def transform(self, XU, Y, labels=None):
        # apply transformation and return transformed copies of data
        return self.transform_x(XU), self.transform_y(Y), labels

    def update_data_config(self, config: load_data.DataConfig):
        """Change the dimensions of the data's configuration to match what the preprocessing will do"""
        pass

    @abc.abstractmethod
    def transform_x(self, XU):
        """Apply transformation on XU"""

    @abc.abstractmethod
    def transform_y(self, Y):
        """Apply transformation on Y"""

    @abc.abstractmethod
    def invert_transform(self, Y, X=None):
        """Invert transformation on Y with potentially information from X (untransformed)"""

    @abc.abstractmethod
    def invert_x(self, X):
        """Invert transformation on X if possible (for transforms that lose information this may be impossible)"""

    @abc.abstractmethod
    def _fit_impl(self, XU, Y, labels):
        """Fit internal state to training set"""


class Compose(Transformer):
    """Compose a list of preprocessors to be applied in order"""

    def __init__(self, transforms: typing.List[Transformer]):
        # order matters here; applied from left to right
        self.transforms = transforms
        super(Compose, self).__init__()

    def transform(self, XU, Y, labels=None):
        # need to override this because held transforms could override these
        for t in self.transforms:
            XU, Y, labels = t.transform(XU, Y, labels)
        return XU, Y, labels

    def _fit_impl(self, XU, Y, labels):
        for t in self.transforms:
            t._fit_impl(XU, Y, labels)
            XU, Y, labels = t.transform(XU, Y, labels)

    def transform_x(self, XU):
        for t in self.transforms:
            XU = t.transform_x(XU)
        return XU

    def transform_y(self, Y):
        for t in self.transforms:
            Y = t.transform_y(Y)
        return Y

    def invert_transform(self, Y, X=None):
        # need to also reverse order
        for t in reversed(self.transforms):
            Y = t.invert_transform(Y, X)
        return Y

    def invert_x(self, X):
        # need to also reverse order
        for t in reversed(self.transforms):
            X = t.invert_x(X)
        return X

    def update_data_config(self, config: load_data.DataConfig):
        for t in self.transforms:
            t.update_data_config(config)


class NoTransform(Transformer):
    def transform_x(self, XU):
        return XU

    def transform_y(self, Y):
        return Y

    def invert_transform(self, Y, X=None):
        return Y

    def invert_x(self, X):
        return X

    def _fit_impl(self, XU, Y, labels):
        # optionally do some fitting on the training set
        pass


class PolynomialState(NoTransform):
    def __init__(self, order=2):
        super().__init__()
        from sklearn.preprocessing import PolynomialFeatures
        self.poly = PolynomialFeatures(order, include_bias=False)

    def _fit_impl(self, XU, Y, labels):
        self.poly.fit(XU.numpy())

    def transform_x(self, XU):
        p = self.poly.transform(XU.numpy())
        return torch.from_numpy(p)

    def invert_x(self, X):
        raise NotImplemented


class SklearnTransformer(Transformer):
    def __init__(self, method, methodY=None):
        super().__init__()
        self.method = method
        self.methodY = methodY or copy.deepcopy(method)

    def _fit_impl(self, XU, Y, labels):
        self.method.fit(XU)
        self.methodY.fit(Y)

    def transform_x(self, XU):
        x = self.method.transform(XU)
        return torch.from_numpy(x)

    def transform_y(self, Y):
        y = self.methodY.transform(Y)
        return torch.from_numpy(y)

    def invert_transform(self, Y, X=None):
        return torch.from_numpy(self.methodY.inverse_transform(Y))

    def invert_x(self, X):
        return torch.from_numpy(self.method.inverse_transform(X))


class SingleTransformer:
    """Transformer like Sklearn's transformers; pytorch version to pass through differentiation"""

    @abc.abstractmethod
    def fit(self, X):
        """Fit to data"""

    @abc.abstractmethod
    def transform(self, X):
        """Transform data based on previous fit"""

    @abc.abstractmethod
    def inverse_transform(self, X):
        """The inverse transformation"""

    def data_dim_change(self):
        """How this transform changed the dimension of the data.

        Also gives enough info to tell if it changed nx or nu

        :return: (change in dimensions, smallest index of dimension created/removed)
        """
        return 0, None


class NullSingleTransformer(SingleTransformer):
    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class SelectTransformer(SingleTransformer):
    """Select some dimensions of incoming data; do nothing on inverse"""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X):
        tensor_utils.ensure_tensor(X.device, torch.long, self.cols)

    def transform(self, X):
        return X[:, self.cols]

    def inverse_transform(self, X):
        return X


class PytorchTransformer(Transformer):
    def __init__(self, method: SingleTransformer, methodY=None):
        super().__init__()
        self.method = method
        self.methodY = methodY or copy.deepcopy(method)

    def _fit_impl(self, XU, Y, labels):
        self.method.fit(XU)
        self.methodY.fit(Y)

    def transform_x(self, XU):
        return self.method.transform(XU)

    def transform_y(self, Y):
        return self.methodY.transform(Y)

    def invert_transform(self, Y, X=None):
        return self.methodY.inverse_transform(Y)

    def invert_x(self, X):
        return self.method.inverse_transform(X)

    def update_data_config(self, config: load_data.DataConfig):
        change, loc = self.method.data_dim_change()
        if loc is not None:
            if loc < config.nx:
                config.nx += change
            else:
                config.nu += change
            if config.n_input:
                config.n_input += change
        change, _ = self.methodY.data_dim_change()
        config.ny += change


class MinMaxScaler(SingleTransformer):
    def __init__(self, feature_range=(0, 1), dims_share_scale: Iterable[typing.List[int]] = (), **kwargs):
        super().__init__(**kwargs)
        self.feature_range = feature_range
        self.dims_share_scale = dims_share_scale
        self._scale, self._min = None, None

    def fit(self, X):
        # TODO assume we have no nans
        self._fit_with_low_high(torch.min(X, dim=0)[0], torch.max(X, dim=0)[0])

    def _fit_with_low_high(self, low, high):
        feature_range = self.feature_range
        nx = low.shape[0]
        # handle per-dimensional feature range
        if type(feature_range[0]) in (list, tuple):
            feature_range = torch.tensor(feature_range)
        if tensor_utils.is_tensor_like(feature_range):
            # should be 2 x nx; could have fewer columns; would fill with (0,1)
            assert len(feature_range) == 2
            assert feature_range.shape[0] == 2
            f = torch.zeros((2, nx), dtype=low.dtype, device=low.device)
            f[1] = 1
            f[:, :feature_range.shape[1]] = feature_range.to(dtype=low.dtype, device=low.device)
            feature_range = f

        data_range = high - low
        # handle zeros/no variation in that dimension
        data_range[data_range == 0.] = 1.
        self._scale = ((feature_range[1] - feature_range[0]) / data_range)
        for shared_groups in self.dims_share_scale:
            self._scale[shared_groups] = self._scale[shared_groups].mean()
            # note that afterward the shared groups will not have max = feature_range[1]
            # but will still have min = feature_range[0]
        self._min = feature_range[0] - low * self._scale

    def transform(self, X):
        return (X * self._scale) + self._min

    def inverse_transform(self, X):
        return (X - self._min) / self._scale


class RobustMinMaxScaler(MinMaxScaler):
    """Like min-max, but ignore outliers"""

    def __init__(self, percentile=0.975, **kwargs):
        super().__init__(**kwargs)
        self.percentile = percentile

    def fit(self, X):
        low = torch.kthvalue(X, max(1, int((1 - self.percentile) * X.shape[0])), dim=0)[0]
        high = torch.kthvalue(X, int(self.percentile * X.shape[0]), dim=0)[0]
        self._fit_with_low_high(low, high)


class StandardScaler(SingleTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._m = None
        self._s = None

    def fit(self, X):
        self._m = X.mean(0, keepdim=True)
        self._s = X.std(0, unbiased=False, keepdim=True)

    def transform(self, X):
        return (X - self._m) / self._s

    def inverse_transform(self, X):
        return (X * self._s) + self._m


class AngleToCosSinRepresentation(SingleTransformer):
    """
    Continuous representation of an angle which allows better approximation by neural networks
    """

    def __init__(self, angle_index):
        # this transform only handles 1 location; for more locations compose them together
        assert type(angle_index) is int
        self.angle_index = angle_index

    def fit(self, X):
        assert self.angle_index < X.shape[1]

    def transform(self, X):
        s = torch.sin(X[:, self.angle_index]).view(-1, 1)
        c = torch.cos(X[:, self.angle_index]).view(-1, 1)
        return torch.cat((X[:, :self.angle_index], s, c, X[:, self.angle_index + 1:]), dim=1)

    def inverse_transform(self, X):
        s = X[:, self.angle_index]
        c = X[:, self.angle_index + 1]
        theta = torch.atan2(s, c)
        return torch.cat((X[:, :self.angle_index], theta.view(-1, 1), X[:, self.angle_index + 2:]), dim=1)

    def data_dim_change(self):
        return 1, self.angle_index


class DatasetPreprocessor(abc.ABC):
    """Wrapper around Transformer for datasets
    with a transformation fitted on the training set and applied to both training and validation set.
    Assumes the dataset is from a dynamical system (X,U,Y,labels)
    """

    def __init__(self, transformer: Transformer):
        self.tsf = transformer

    def fit(self, dataset):
        if self.tsf.fitted:
            logger.warning("Ignoring attempt to refit preprocessor")
            return
        XU, Y, labels = dataset[:]
        self.tsf.fit(XU, Y, labels)

    def transform(self, dataset):
        if len(dataset):
            XU, Y, labels = self.tsf.transform(*dataset[:])
        else:
            XU, Y, labels = (torch.tensor([], device=series.device) for series in dataset[:])
        return load_data.SimpleDataset(XU, Y, labels)

    def update_data_config(self, config: load_data.DataConfig):
        return self.tsf.update_data_config(config)

    def transform_x(self, XU):
        return self.tsf.transform_x(XU)

    def transform_y(self, Y):
        return self.tsf.transform_y(Y)

    def invert_transform(self, Y, X=None):
        return self.tsf.invert_transform(Y, X)

    def invert_x(self, X):
        return self.tsf.invert_x(X)

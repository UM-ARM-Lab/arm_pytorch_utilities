import abc
import copy
import logging
import typing

import torch
from arm_pytorch_utilities import load_data

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

    def update_data_config(self, config: load_data.DataConfig):
        for t in self.transforms:
            t.update_data_config(config)


class NoTransform(Transformer):
    def transform_x(self, XU):
        return XU

    def transform_y(self, Y):
        return Y

    def invert_transform(self, Y, X=None):
        raise NotImplemented

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


class NullSingleTransformer(SingleTransformer):
    def fit(self, X):
        pass

    def transform(self, X):
        return X

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


class MinMaxScaler(SingleTransformer):
    def __init__(self, feature_range=(0, 1), **kwargs):
        super().__init__(**kwargs)
        self.feature_range = feature_range
        self._scale, self._min = None, None

    def fit(self, X):
        feature_range = self.feature_range
        # TODO assume we have no nans for now
        data_min = torch.min(X, dim=0)[0]
        data_max = torch.max(X, dim=0)[0]

        data_range = data_max - data_min
        # handle zeros/no variation in that dimension
        data_range[data_range == 0.] = 1.
        self._scale = ((feature_range[1] - feature_range[0]) / data_range)
        self._min = feature_range[0] - data_min * self._scale

    def transform(self, X):
        return (X * self._scale) + self._min

    def inverse_transform(self, X):
        return (X - self._min) / self._scale


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


class DatasetPreprocessor(abc.ABC):
    """Pre-process the entire dataset
    with a transformation fitted on the training set and applied to both training and validation set.
    Assumes the dataset is from a dynamical system (X,U,Y,labels)
    """

    def __init__(self, transformer: Transformer):
        self.tsf = transformer

    def fit(self, dataset):
        if self.tsf.fitted:
            logger.warning("Ignoring attempt to refit preprocessor")
            return
        XU, Y, labels = load_data.get_all_data_from_dataset(dataset)
        self.tsf.fit(XU, Y, labels)

    def transform(self, dataset):
        XU, Y, labels = load_data.get_all_data_from_dataset(dataset)
        XU, Y, labels = self.tsf.transform(XU, Y, labels)
        return load_data.SimpleDataset(XU, Y, labels)

    def update_data_config(self, config: load_data.DataConfig):
        return self.tsf.update_data_config(config)

    def transform_x(self, XU):
        return self.tsf.transform_x(XU)

    def transform_y(self, Y):
        return self.tsf.transform_y(Y)

    def invert_transform(self, Y, X=None):
        return self.tsf.invert_transform(Y, X)

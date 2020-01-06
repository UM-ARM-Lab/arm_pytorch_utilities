import abc

from arm_pytorch_utilities import load_data
import copy
import torch
import logging

logger = logging.getLogger(__name__)


class Preprocess(abc.ABC):
    """Pre-process the entire dataset
    with an operation fitted on the training set and applied to both training and validation set.
    Assumes the dataset is from a dynamical system (X,U,Y,labels)

    Different from pytorch's transforms as those are typically applied as you retrieve the data (usually images)
    """

    def __init__(self, strip_affine=False):
        self.strip_affine = strip_affine
        # allow only fit once; for refit, use a different instance of the preprocessor
        # usually refits are mistakes
        self.fitted = False

    def fit(self, dataset):
        if self.fitted:
            logger.warning("Ignoring attempt to refit preprocessor")
            return
        XU, Y, labels = load_data.get_all_data_from_dataset(dataset)
        # strip last affine column
        if self.strip_affine:
            XU = XU[:, :-1]
        self._fit_impl(XU, Y, labels)
        self.fitted = True

    def transform(self, dataset):
        XU, Y, labels = load_data.get_all_data_from_dataset(dataset)
        if self.strip_affine:
            XU = XU[:, :-1]
        XU, Y, labels = self._transform_impl(XU, Y, labels)
        if self.strip_affine:
            XU = torch.cat((XU, XU[:, -1].view(-1, 1)), dim=1)
        return load_data.SimpleDataset(XU, Y, labels)

    @abc.abstractmethod
    def transform_x(self, XU):
        """Apply transformation on XU"""

    @abc.abstractmethod
    def transform_y(self, Y):
        """Apply transformation on Y"""

    @abc.abstractmethod
    def invert_transform(self, y):
        """Invert transformation on Y"""

    @abc.abstractmethod
    def _fit_impl(self, XU, Y, labels):
        """Fit internal state to training set"""

    def _transform_impl(self, XU, Y, labels):
        # apply transformation and return transformed copies of data
        return self.transform_x(XU), self.transform_y(Y), labels


class NoTransform(Preprocess):
    def transform_x(self, XU):
        return XU

    def transform_y(self, Y):
        return Y

    def invert_transform(self, y):
        raise NotImplemented

    def _fit_impl(self, XU, Y, labels):
        # optionally do some fitting on the training set
        pass


class PolynomialState(NoTransform):
    def __init__(self, order=2, **kwargs):
        super().__init__(**kwargs)
        from sklearn.preprocessing import PolynomialFeatures
        self.poly = PolynomialFeatures(order, include_bias=False)

    def _fit_impl(self, XU, Y, labels):
        self.poly.fit(XU.numpy())

    def transform_x(self, XU):
        p = self.poly.transform(XU.numpy())
        return torch.from_numpy(p)


class SklearnPreprocessing(Preprocess):
    def __init__(self, method, methodY=None, **kwargs):
        super().__init__(**kwargs)
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

    def invert_transform(self, y):
        return torch.from_numpy(self.methodY.inverse_transform(y))


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


class PytorchPreprocessing(Preprocess):
    def __init__(self, method: SingleTransformer, methodY=None, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.methodY = methodY or copy.deepcopy(method)

    def _fit_impl(self, XU, Y, labels):
        self.method.fit(XU)
        self.methodY.fit(Y)

    def transform_x(self, XU):
        return self.method.transform(XU)

    def transform_y(self, Y):
        return self.methodY.transform(Y)

    def invert_transform(self, y):
        return self.methodY.inverse_transform(y)


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

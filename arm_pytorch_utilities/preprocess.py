import abc

from arm_pytorch_utilities import load_data as load_utils
from hybrid_sysid import load_data
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
        XU, Y, labels = load_data.get_states_from_dataset(dataset)
        # strip last affine column
        if self.strip_affine:
            XU = XU[:, :-1]
        self._fit_impl(XU, Y, labels)
        self.fitted = True

    def transform(self, dataset):
        XU, Y, labels = load_data.get_states_from_dataset(dataset)
        if self.strip_affine:
            XU = XU[:, :-1]
        XU, Y, labels = self._transform_impl(XU, Y, labels)
        if self.strip_affine:
            XU = torch.cat((XU, XU[:, -1].view(-1, 1)), dim=1)
        return load_utils.SimpleDataset(XU, Y, labels)

    def transform_x(self, XU):
        return XU

    def transform_y(self, Y):
        return Y

    def invert_transform(self, y):
        raise NotImplemented

    def _fit_impl(self, XU, Y, labels):
        # optionally do some fitting on the training set
        pass

    def _transform_impl(self, XU, Y, labels):
        # apply transformation and return transformed copies of data
        return self.transform_x(XU), self.transform_y(Y), labels


class PolynomialState(Preprocess):
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

    def _transform_impl(self, XU, Y, labels):
        return self.transform_x(XU), self.transform_y(Y), labels

    def invert_transform(self, y):
        return torch.from_numpy(self.methodY.inverse_transform(y))

import torch
import matplotlib.pyplot as plt
from arm_pytorch_utilities.make_data import make, select
import numpy as np


class DataSet:
    def __init__(self, N=200, variance=0.1, input_dim=3, output_dim=1, num_modes=3, selector=None,
                 use_gpu_if_available=False):
        self.N = N
        self.variance = variance

        # GPU speedup
        if use_gpu_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._p = input_dim
        self._n = output_dim
        self._s = num_modes
        self._H = 1

        self._pp = (torch.rand((self._s, self._p, self._n), dtype=torch.double) - 0.5) * 4
        self._train = None
        self._val = None
        self._selector = selector
        self._tsf = None

    def training_set(self):
        return self._train

    def validation_set(self):
        return self._val

    def feature_transformation(self):
        """Ground truth function to transform input to feature"""
        return self._tsf

    def create_feature_transformation(self):
        return None

    def make_data(self):
        """Separate function from constructor to allow reseeding; assumes transformation has been created"""
        if self._tsf is None:
            self._tsf = self.create_feature_transformation()

        if self._selector is None:
            self._selector = select.RandomSelector(self._tsf, 0.2)

        self._train = make.make(self._pp, self.N, self.variance, selector=self._selector, sorted=True)
        self._val = make.make(self._pp, self.N, self.variance, selector=self._selector, sorted=True)


class LinearDataSet(DataSet):
    """Feature is a linear combination of the inputs"""

    def __init__(self, **kwargs):
        super(LinearDataSet, self).__init__(**kwargs)
        self.make_data()

    def create_feature_transformation(self):
        linear = torch.nn.Sequential(
            torch.nn.Linear(self._p, self._H, bias=False),
        ).double().to(self.device)
        linear[0].weight.data = (torch.rand((1, self._p), dtype=torch.double) - 0.5) * 6
        return linear

    def plot_training(self):
        U, Y, labels = self._train
        plt.figure()
        for i in range(self._s):
            in_cluster = labels == i
            plt.scatter(U[in_cluster, 0], Y[in_cluster])


class PolynomialDataSet(DataSet):
    """2D input 1D output, mode is separable in x1^2 + x2^2"""

    def __init__(self, order=2, target_params=None, polynomial_bias=False, **kwargs):
        super(PolynomialDataSet, self).__init__(**kwargs)

        from sklearn.preprocessing import PolynomialFeatures
        self._order = order
        self._poly = PolynomialFeatures(self._order, include_bias=polynomial_bias)
        # create input sample to fit (tells sklearn what input sizes to expect)
        u = np.random.rand(2).reshape(1, -1)
        self._poly.fit(u)

        target = target_params
        if target is None:
            # fixed weight to ensure x1^2 + x2^2 is the feature
            if self._order is 2:
                target = [0, 0, 1, 0, 1]
            elif self._order is 3:
                target = [0, 0, 1, 0, 1, 0, 0, 0, 0]
            else:
                raise RuntimeError("Unhandled polynomial order: {}".format(self._order))
            if polynomial_bias:
                target += [0]
        self._target_params = torch.tensor([target], dtype=torch.double, device=self.device)
        assert self._poly.n_output_features_ == self._target_params.shape[1]

        self.make_data()

    def create_feature_transformation(self):
        linear = torch.nn.Sequential(
            torch.nn.Linear(self._poly.n_output_features_, self._H, bias=False),
        ).double().to(self.device)

        linear[0].weight.data = self._target_params.clone()

        return self._tsf_from_ltsf(linear)

    def _tsf_from_ltsf(self, ltsf):
        def tsf(xu):
            # only polynomials on x1 and x2
            x = xu[:, :2].cpu().numpy()
            polyout = self._poly.transform(x)
            xx = torch.from_numpy(polyout).to(self.device)
            features = ltsf(xx)
            return features

        return tsf

    def plot_training(self):
        from mpl_toolkits.mplot3d import Axes3D
        U, Y, labels = self._train

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        symbols = ['x', 'o', '*']
        for j in range(self._s):
            in_cluster = labels == j
            ax.scatter(U[in_cluster, 0], U[in_cluster, 1], Y[in_cluster], marker=symbols[j % len(symbols)])

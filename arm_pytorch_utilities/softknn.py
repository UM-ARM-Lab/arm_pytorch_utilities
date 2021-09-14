import torch


class SoftKNN(torch.nn.Module):
    """
    Differentiable relaxation of the k-nearest neighbour operation.
    For a N x n feature input, we output a N x N weight matrix such that the (i,j) element is the weight of the
    jth data point of being one of the k nearest neighbours of the ith data point. Weights are non-negative and
    their upper value range depends on initialization options.
    """

    def __init__(self, min_k=5, activation='linear', uniform_threshold=False, normalization=None):
        """
        :param min_k: Minimum number of neighbours to consider at each point; the behaviour depends on what activation
        is used and whether a uniform threshold is chosen
        :param activation: The function to apply to the shifted distances to get positive weights; currently there is
        'linear' which is ReLU and otherwise we use sigmoid. We recover hard KNN by using the sign activation function
        (1 for distance >= 0 and 0 otherwise).
        :param uniform_threshold: Whether a single distance threshold should be chosen for all data points such that
        at least min_k points is guaranteed to have positive weights for every data point
        :param normalization: None or the p value for the L_p norm, use 1 for sum to 1, use 2 for norm to 1
        """
        super(SoftKNN, self).__init__()
        self.min_neighbours = min_k
        self.activation = activation
        self.normalization = normalization
        self.uniform_threshold = uniform_threshold

    def forward(self, features):
        # calculate the distances of every point to every other point in feature space
        dists = torch.cdist(features, features)

        # select a distance threshold such that any point beyond this is weighted 0
        # basically the max of the kth smallest element inside each row
        dd, ddi = torch.topk(dists, self.min_neighbours + 1, largest=False, sorted=False, dim=1)
        threshold = dd[:, self.min_neighbours]

        if self.uniform_threshold:
            threshold = torch.max(threshold)
            dd = (threshold - dists)
        else:
            dd = (threshold[:, None] - dists)

        # avoid edge case of multiple elements at kth closest distance causing them to become 0
        dd += 1e-10

        # TODO add more activation options
        # apply weighting function to the distances/nodes
        if self.activation == 'linear':
            weights = dd.clamp(min=0)
        else:
            weights = torch.sigmoid(dd * self.activation)

        # normalization
        if self.normalization is not None:
            weights = torch.nn.functional.normalize(weights, dim=1, p=self.normalization)

        return weights

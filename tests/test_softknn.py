"""Verify that our soft KNN method passes through gradients and is able to learn parameters through it"""
import logging

import torch
from arm_pytorch_utilities import load_data, rand
from arm_pytorch_utilities import softknn
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


# generate ground truth transforms and operations
# converts it to a transformed space
class SimpleNet(torch.nn.Module):
    def __init__(self, D_in, H):
        super(SimpleNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=False)
        self.knn = softknn.SoftKNN(min_k=20)

    def forward(self, x, y):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        features = self.linear1(x)

        weights = self.knn(features)
        # TODO weighted least squares in x-y space rather than feature space using these weights
        # propagate gradient backwards to transform parameters through these weights
        # TODO remove: for now can just sum up xs inside the neighbourhood

        output = torch.zeros_like(x)
        for i, w in enumerate(weights):
            # nw = w / torch.sum(w)
            # output[i] = torch.matmul(nw, x)
            # can drop out the terms that have 0 weight and seeing if it will affect differentiability
            neighbours = torch.nonzero(w).view(-1)
            nw = w[neighbours]
            nw /= torch.sum(nw)
            output[i] = torch.matmul(nw, x[neighbours])

        return output


def KNN(features, k):
    # features = features.float()
    dist_mat = torch.cdist(features, features)
    # ith row are the k nearest neighbours of ith data point
    dists, Idx = torch.topk(dist_mat, k, largest=False, sorted=False, dim=1)
    return dists, Idx


def test_softknn(debug=False):
    # doesn't always converge in time for all random seed
    seed = 318455
    logger.info('random seed: %d', rand.seed(seed))

    D_in = 3
    D_out = 1

    target_params = torch.rand(D_in, D_out).t()
    # target_params = torch.tensor([[1, -1, 1]], dtype=torch.float )
    target_tsf = torch.nn.Linear(D_in, D_out, bias=False)
    target_tsf.weight.data = target_params
    for param in target_tsf.parameters():
        param.requires_grad = False

    def produce_output(X):
        # get the features
        y = target_tsf(X)
        # cluster in feature space
        dists, Idx = KNN(y, 5)

        # take the sum inside each neighbourhood
        # TODO do a least square fit over X inside each neighbourhood
        features2 = torch.zeros_like(X)
        for i in range(dists.shape[0]):
            # md = max(dists[i])
            # d = md - dists[i]
            # w = d / torch.norm(d)
            features2[i] = torch.mean(X[Idx[i]], 0)
            # features2[i] = torch.matmul(w, X[Idx[i]])

        return features2

    N = 400
    ds = load_data.RandomNumberDataset(produce_output, num=400, input_dim=D_in)
    train_set, validation_set = load_data.split_train_validation(ds)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=N, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=N, shuffle=False)

    criterion = torch.nn.MSELoss(reduction='sum')

    model = SimpleNet(D_in, D_out)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    losses = []
    vlosses = []
    pdist = []
    cosdist = []

    def evaluateLoss(data):
        # target
        x, y = data
        pred = model(x, y)

        loss = criterion(pred, y)
        return loss

    def evaluateValidation():
        with torch.no_grad():
            loss = sum(evaluateLoss(data) for data in val_loader)
            return loss / len(val_loader.dataset)

    # model.linear1.weight.data = target_params.clone()
    for epoch in range(200):
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()

            loss = evaluateLoss(data)
            loss.backward()
            optimizer.step()

            avg_loss = loss.item() / len(data[0])

            losses.append(avg_loss)
            vlosses.append(evaluateValidation())
            pdist.append(torch.norm(model.linear1.weight.data - target_params))
            cosdist.append(torch.nn.functional.cosine_similarity(model.linear1.weight.data, target_params))
            if debug:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, avg_loss))

    if debug:
        print('Finished Training')
        print('Target params: {}'.format(target_params))
        print('Learned params:')
        for param in model.parameters():
            print(param)

        print('validation total loss: {:.3f}'.format(evaluateValidation()))

    model.linear1.weight.data = target_params.clone()
    target_loss = evaluateValidation()

    if debug:
        print('validation total loss with target params: {:.3f}'.format(target_loss))

        plt.plot(range(len(losses)), losses)
        plt.plot(range(len(losses)), vlosses)
        plt.plot(range(len(losses)), [target_loss] * len(losses), linestyle='--')
        plt.legend(['training minibatch', 'whole validation', 'validation with target params'])
        plt.xlabel('minibatch')
        plt.ylabel('MSE loss')

        plt.figure()
        plt.plot(range(len(pdist)), pdist)
        plt.xlabel('minibatch')
        plt.ylabel('euclidean distance of model params from target')

        plt.figure()
        plt.plot(range(len(cosdist)), cosdist)
        plt.xlabel('minibatch')
        plt.ylabel('cosine similarity between model params and target')
        plt.show()

    # check that we're close to the actual KNN performance on validation set
    last_few = 5
    loss_tolerance = 0.02
    assert sum(vlosses[-last_few:]) / last_few - target_loss < target_loss * loss_tolerance


if __name__ == "__main__":
    test_softknn(True)

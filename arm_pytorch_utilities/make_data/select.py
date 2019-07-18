import torch


def identity_select(xu):
    labels = xu.new_empty(xu.shape[0], dtype=torch.long)
    u = xu[:, 0]
    labels[u <= -1] = 0
    labels[(u >= -1) & (u <= 2)] = 1
    labels[u > 2] = 2
    return labels


class RandomSelector:
    def __init__(self, tsf, min_proportion_each_class):
        self.tsf = tsf
        # min proportion of each data point
        self.min_proportion_each_class = min_proportion_each_class
        self.thresholds = None

    def __call__(self, xu):
        N = xu.shape[0]
        labels = xu.new_empty(N, dtype=torch.long)

        with torch.no_grad():
            u = self.tsf(xu)
            u = u.view(-1)
            if self.thresholds is None:
                self._calculate_thresholds(u)

            categories = self._calculate_categories(u)
            for i, cat in enumerate(categories):
                labels[cat] = i

        return labels

    def _calculate_thresholds(self, u):
        N = u.shape[0]
        rr = (torch.min(u), torch.max(u))

        bad_proportion = True
        while bad_proportion:
            bad_proportion = False
            thresholds = torch.rand(2, dtype=u.dtype, device=u.device) * (rr[1] - rr[0]) + rr[0]
            self.thresholds = torch.sort(thresholds)[0]

            categories = self._calculate_categories(u)
            for i, cat in enumerate(categories):
                n = torch.sum(cat)
                # require a minimum proportion of each data point
                if n < N * self.min_proportion_each_class:
                    bad_proportion = True
                    break

    def _calculate_categories(self, u):
        categories = [u <= self.thresholds[0], (u >= self.thresholds[0]) & (u <= self.thresholds[1]),
                      u > self.thresholds[1]]
        return categories

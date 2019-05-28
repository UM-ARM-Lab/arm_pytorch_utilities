import torch
import torch.utils.data


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
        return tuple(sequence[idx] for sequence in self.sequences)


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


def splitTrainValidationSets(dataset, validation_ratio=0.1):
    # consider giving a shuffle (with np.random.shuffle()) option to permute the data before viewing
    offset = int(len(dataset) * (1 - validation_ratio))
    return PartialViewDataset(dataset, 0, offset), PartialViewDataset(dataset, offset, len(dataset) - offset)

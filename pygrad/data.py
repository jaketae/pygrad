import warnings

import numpy as np


def _check_ratio(ratio):
    if not (0 <= ratio <= 1):
        raise ValueError(f"ratio must be between 0 and 1, but received {ratio}")


def ratio_split(dataset, *ratios):
    if isinstance(ratios[0], tuple):
        ratios = ratios[0]
    for ratio in ratios:
        _check_ratio(ratio)
    if sum(ratios) != 1:
        warnings.warn("some data may be lost since ratio does not sum up to 1")
    splits = []
    start = 0
    index = np.random.permutation(np.arange(len(dataset)))
    for ratio in ratios:
        end = start + int(ratio * len(dataset))
        splits.append(Subset(dataset, index[start:end]))
        start = end
    return splits


class Dataset:
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x
        self.data = None
        self.label = None

    def __getitem__(self, index):
        if self.label is None:
            return self.transform(self.data[index]), None
        return (
            self.transform(self.data[index]),
            self.target_transform(self.label[index]),
        )

    def __len__(self):
        return len(self.data)


class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iter = 0
        self.max_iter = len(dataset) // batch_size
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        i = self.iter
        if i == self.max_iter:
            self.reset()
            raise StopIteration
        batch_size = self.batch_size
        batch_index = self.index[i * batch_size : (i + 1) * batch_size]
        batch = self.dataset[batch_index]
        self.iter += 1
        return batch

    def reset(self):
        self.iter = 0
        if self.shuffle:
            self.index = np.random.permutation(np.arange(len(self.dataset)))
        else:
            self.index = np.arange(len(self.dataset))

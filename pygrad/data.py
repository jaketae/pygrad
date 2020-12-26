import numpy as np


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self, index):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iter = 0
        self.max_iter = len(dataset) // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        i = self.iter
        if i == self.max_iter:
            self.reset()
        batch_size = self.batch_size
        batch_index = self.index[i * batch_size : (i + 1) * batch_size]
        batch = self.dataset[batch_index]
        self.iter += 1
        return batch[:, 0], batch[:, 1]

    def reset(self):
        self.iter = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

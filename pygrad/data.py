class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self, index):
        pass


class DataLoader:
    pass

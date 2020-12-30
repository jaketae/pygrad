# Data

PyGrad offers basic data processing APIs that allow for easy creation of `Dataset` and `DataLoader` objects. 

## Dataset

PyGrad `Dataset` class adds a generic yet useful layer of abstraction that allows it to consume datasets that come in all shapes and forms. Specifically, all that has to be specified is the `data` and `label` attribute for each `Dataset` object. In the context of unsupervised learning, `label` can be set to `None`.

Below, we provide an example of creating a custom dataset using `sklearn`'s `make_moons()` function.

```python
from sklearn import datasets as skds

class MoonDataset(data.Dataset):
    def __init__(self, num_samples, noise=0.1, *args, **kwargs):
        super(MoonDataset, self).__init__(*args, **kwargs)
        X, y = skds.make_moons(num_samples, noise=noise, shuffle=True)
        self.data = X
        self.label = y
```

`Dataset` objects can easily be split up via the `ratio_split()` function, which can particularly be useful for creating training, validation, and test sets. 

```python
from pygrad.data import ratio_split

dataset = MoonDataset()
train_ds, test_ds = ratio_split(dataset, 0.8, 0.2)
```

## DataLoader

PyGrad's `DataLoader` class allows basic batching and shuffling functionality to be applied to `pygrad.data.Dataset` instances. 

```python
from pygrad.data import DataLoader

BATCH_SIZE = 8
train_loader = DataLoader(train_ds, BATCH_SIZE)
test_loader = DataLoader(test_ds, BATCH_SIZE)
```

`DataLoader` instances can be iterated as follows:

```python
for data, labels in train_loader:
    # training logic here
```
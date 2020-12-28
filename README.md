# PyGrad

PyGrad is a [NumPy](https://numpy.org)-based pure Python mini automatic differentiation library. Adopting a define-by-run approach, PyGrad provides a clear, intuitive interface for constructing computational graphs, calculating gradients, and solving optimization problems. Building on top of these strengths as an autograd library, PyGrad also offers a minimal neural network API, inspired heavily by [PyTorch](https://pytorch.org). 

## Installation

PyGrad is available on PyPi. 

```
pip install pygrad
```

To build and develop from source, clone this repository via

```
git clone https://github.com/jaketae/pygrad.git
```

## Quick Start

Full documentation to be made available at [PyGrad docs](https://pygrad.readthedocs.io/en/latest/).

### Variable Class

PyGrad's main data class is `Variable`, through which computational graphs can be created. 

```python
>>> from pygrad import Variable
>>> a = Variable(5)
>>> b = Variable(3)
>>> a + b
Variable(8)
```

PyGrad can, of course, also deal with arrays and tensors.

```python
>>> m1 = Variable([[1, 2], [3, 4]])
>>> m2 = Variable([[1, 1], [1, 1]])
>>> m1 + m2
Variable([[2 3]
          [4 5]])
```

Since PyGrad uses NumPy as its backend, it supports broadcasting as well as other matrix operations including transpose, reshape, and matrix multiplication. 

```python
>>> m1.T
Variable([[1 3]
          [2 4]])
>>> m1.reshape(1, 4)
Variable([[1 2 3 4]])
```

### Gradient Computation

Under the hood, PyGrad creates computational graphs on the fly when operations are performed on a `Variable` instance. To obtain gradient values, simply call the `backward()` method on a leaf variable.

```python
>>> x = Variable(3)
>>> y = x * x
>>> y.backward()
>>> x.grad
Variable(6)
```

By default, PyGrad deletes gradients for intermediary variables to efficiently utilize available memory. To keep the gradient values for all variables involved in the computation graph, set `retain_grad=True` in the `backward()` call. In the example below, we calculate <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20z%3D2x%5E2" alt="z=2x^2">, where <img src="https://latex.codecogs.com/svg.latex?%5Clarge%20y%3D2x"> is an intermediary variable.

```python
>>> x = Variable(3)
>>> y = x * x
>>> z = y + y
>>> z.backward(retain_grad=True)
>>> z.grad
Variable(1)
>>> y.grad
Variable(2)
>>> x.grad
Variable(12)
```

## Neural Networks

PyGrad offers an intuitive interface for building and training neural networks. Specifically, ANN-related components live in the `pygrad.nn` module. For those who are already familiar with PyTorch will immediately see that PyGrad's API is no different from that of PyTorch. We hope to add more flavor to PyGrad's API in the days to come.

### Network Initialization

Below is a simple example of how to declare a neural network in PyGrad.

```python
from pygrad import nn
from pygrad import functions as F

class NeuralNet(nn.Module):
    def __init__(
        self, num_input, num_hidden, num_class, dropout
    ):
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_class)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
```

`pygrad.nn` includes layers and the `Module` class through which neural networks can be initialized. Also noteworthy is `pygrad.functions`, which includes activation functions, trigonometric functions, as well as a host of other basic operations such as reshape and transpose. 

### DataLoader Class

PyGrad's `DataLoader` class allows basic batching and shuffling functionality to be applied to `pygrad.data.Dataset` instances. Here, we assume a custom `Dataset` instance, called `CatDogDataset`. 

```python
from pygrad.data import DataLoader, ratio_split

BATCH_SIZE = 8

dataset = CatDogDataset()
train_ds, test_ds = ratio_split(dataset, 0.8, 0.2)
train_loader = DataLoader(train_ds, BATCH_SIZE)
test_loader = DataLoader(test_ds, BATCH_SIZE)
```

 `DataLoader` instances can be iterated as follows:

```python
for data, labels in train_loader:
    # training logic here
```

### Optimizers

PyGrad offers a number of different optimizers. These include

* `pygrad.nn.optim.SGD`
* `pygrad.nn.optim.AdaGrad`
* `pygrad.nn.optim.AdaDelta`
* `pygrad.nn.optim.Adam`

Training a model with optimizers can be done by instantiating an `nn.optim.Optimizer` object with some model's parameters. After calling `backward()` on a loss value, simply call `step()` on the optimizer to update the target model's parameters.

```python
# imports assumed

optimizer = nn.optim.Adam(model.params())

for data, labels in train_loader:
    pred = model(data)
    loss = F.softmax_cross_entropy(pred, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The optimizer will update the model's weights according to the gradient values of each parameter. 

## Credits

PyGrad is heavily based upon [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3/tree/master/dezero), an educational library introduced in [Deep Learning from Scratch 3](https://koki0702.github.io/dezero-book/en/index.html). Much of PyGrad's initial code base was adapted from DeZero. The design language of PyGrad's neural network API was inspired by and borrowed from PyTorch. [Chainer](https://chainer.org) is also worthy of mention as well, as DeZero itself also adapted many features from Chainer. Last but not least, PyGrad would not have been made possible without NumPy. Our acknowledgement goes to all the developers who put their time and effort into developing the aforementioned libraries. 

## License

Released under the [MIT License](LICENSE.md).
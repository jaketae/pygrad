import weakref

import numpy as np

import pygrad.functions as F
from pygrad import utils
from pygrad.core import Parameter, as_tuple


class Module:
    def __init__(self):
        self._params = set()
        self.is_train = True

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Module)):
            self._params.add(name)
        super(Module, self).__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = as_tuple(self.forward(*inputs))
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        if len(outputs) > 1:
            return outputs
        return outputs[0]

    def __repr__(self):
        return str(self.__dict__)

    def forward(self, inputs):
        raise NotImplementedError

    def params(self):
        for name in self._params:
            param = self.__dict__[name]
            if isinstance(param, Module):
                yield from param.params()
            else:
                yield param

    def train(self, is_train=True):
        self.is_train = is_train
        for name in self._params:
            param = self.__dict__[name]
            if isinstance(param, Module):
                param.train(is_train)

    def eval(self):
        self.train(False)

    def plot(self, to_file="graph.png", dpi=300):
        try:
            self.inputs
        except AttributeError:
            raise RuntimeError("need to run a forward pass first")
        utils.plot_model(self, to_file, dpi)

    def weight_dict(self):
        weights = {}
        for name in self._params:
            param = self.__dict__[name]
            if isinstance(param, Module):
                weights[name] = param.weight_dict()
            else:
                weights[name] = param.data
        return weights

    def load_weight_dict(self, weights):
        for name, weight in weights.items():
            param = self.__dict__[name]
            if isinstance(param, Module):
                if not isinstance(weight, dict):
                    raise KeyError(
                        "cannot load model due to missing or mismatching keys"
                    )
                param.load_weight_dict(weight)
            else:
                if param.data.shape != weight.shape:
                    raise KeyError(
                        "cannot load model due to missing or mismatching keys"
                    )
                param.data = weight

    def load(self, path):
        weights = np.load(path, allow_pickle=True).item()
        self.load_weight_dict(weights)

    def save(self, path):
        weights = self.weight_dict()
        np.save(path, weights)


class Linear(Module):
    def __init__(self, in_size, out_size, bias=True):
        super(Linear, self).__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name="W")
        if bias:
            self.b = Parameter(np.zeros(out_size), name="b")
        else:
            self.b = None

    def forward(self, x):
        return F.linear(x, self.W, self.b)


class Dropout(Module):
    def __init__(self, dropout=0.5):
        super(Dropout, self).__init__()
        self.dropout = Parameter(dropout)

    def forward(self, x):
        return F.dropout(x, self.dropout, self.is_train)

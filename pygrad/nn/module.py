import weakref
from typing import Dict, Generator, Tuple, Union

import numpy as np

import pygrad.functions as F
from pygrad import utils
from pygrad.core import Optional, Parameter, Variable, as_tuple

WeightDictType = Dict[str, Union[dict, np.ndarray]]


class Module:
    def __init__(self):
        self._params = set()
        self.is_train = True

    def __setattr__(self, name: str, value: Union[bool, Optional[Parameter]]) -> None:
        if isinstance(value, (Parameter, Module)):
            self._params.add(name)
        super(Module, self).__setattr__(name, value)

    def __call__(self, *inputs) -> Union[Variable, Tuple[Variable, ...]]:
        outputs = as_tuple(self.forward(*inputs))
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        if len(outputs) > 1:
            return outputs
        return outputs[0]

    def __repr__(self) -> str:
        return str(self.__dict__)

    def forward(self, inputs):
        raise NotImplementedError

    def params(self) -> Generator[Parameter, None, None]:
        for name in self._params:
            param = self.__dict__[name]
            if isinstance(param, Module):
                yield from param.params()
            else:
                yield param

    def train(self, is_train: bool = True) -> None:
        self.is_train = is_train
        for name in self._params:
            param = self.__dict__[name]
            if isinstance(param, Module):
                param.train(is_train)

    def eval(self) -> None:
        self.train(False)

    def plot(self, to_file: str = "graph.png", dpi: int = 300) -> None:
        try:
            self.inputs
        except AttributeError:
            raise RuntimeError("need to run a forward pass first")
        utils.plot_model(self, to_file, dpi)

    def weight_dict(self) -> WeightDictType:
        weights = {}
        for name in self._params:
            param = self.__dict__[name]
            if isinstance(param, Module):
                weights[name] = param.weight_dict()
            else:
                weights[name] = param.data
        return weights

    def load_weight_dict(self, weights: WeightDictType) -> None:
        for name, weight in weights.items():
            param = self.__dict__[name]
            if isinstance(param, Module):
                if not isinstance(weight, dict):
                    raise KeyError(
                        "cannot load model due to missing or mismatching keys"
                    )
                param.load_weight_dict(weight)
            elif isinstance(weight, np.ndarray):
                if param.data.shape != weight.shape:
                    raise KeyError(
                        "cannot load model due to missing or mismatching keys"
                    )
                param.data = weight

    def load(self, path: str) -> None:
        weights = np.load(path, allow_pickle=True).item()
        self.load_weight_dict(weights)

    def save(self, path: str) -> None:
        weights = self.weight_dict()
        np.save(path, weights)


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias=True):
        super(Linear, self).__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name="W")
        self.b: Optional[Parameter] = None
        if bias:
            self.b = Parameter(np.zeros(out_size), name="b")

    def forward(self, x):
        return F.linear(x, self.W, self.b)


class Dropout(Module):
    def __init__(self, dropout=0.5):
        super(Dropout, self).__init__()
        self.dropout = Parameter(dropout)

    def forward(self, x):
        return F.dropout(x, self.dropout, self.is_train)

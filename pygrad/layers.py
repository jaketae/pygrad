import weakref

import pygrad.functions as F
from pygrad.core import Parameter, as_tuple


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super(Layer, self).__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = as_tuple(self.forward(*inputs))
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        if len(outputs) > 1:
            return outputs
        return outputs[0]


    def forward(self, inputs):
        raise NotImplementedError

    def params(self):
        for param in self._params:
            yield self.__dict__[param]

    def clear_grads(self):
        for param in self._params:
            param.clear_grad()

import weakref

import numpy as np

import pygrad.functions as F
from pygrad.core import Parameter, as_tuple


class Module:
    def __init__(self):
        self._params = set()

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
        for param in self._params:
            obj = self.__dict__[param]
            if isinstance(obj, Module):
                yield from obj.params()
            yield obj

    def clear_grads(self):
        for param in self._params:
            param.clear_grad()

    


class Linear(Module):
    def __init__(self, in_size, out_size, bias=True, dtype=np.float32):
        super(Linear, self).__init__()
        self.W = Parameter(np.random.randn(in_size, out_size).astype(dtype), name="W")
        if bias:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")
        else:
            self.b = None

    def forward(self, x):
        return F.linear(x, self.W, self.b)

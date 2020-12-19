import heapq

import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func, return_var=False):
        self.creator = func
        self.generation = func.generation + 1
        if return_var:
            return self

    def clear_grad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        seen_set = set(funcs)
        while funcs:
            f = heapq.heappop(funcs)
            gys = [output.grad for output in f.outputs]
            gxs = as_tuple(f.backward(*gys))
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if not (x.creator is None or x.creator in seen_set):
                    seen_set.add(x.creator)
                    heapq.heappush(funcs, x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = as_tuple(self.forward(*xs))
        self.inputs = inputs
        self.generation = max([x.generation for x in inputs])
        outputs = [Variable(as_array(y)).set_creator(self, return_var=True) for y in ys]
        self.outputs = outputs
        if len(outputs) > 1:
            return outputs
        return outputs[0]

    def __lt__(self, other):
        return self.generation > other.generation

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


def as_array(x):
    if np.isscalar(x):
        return np.asarray(x)
    return x


def as_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x

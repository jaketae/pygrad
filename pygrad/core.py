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

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        seen_set = set([self.creator])
        while funcs:
            f = heapq.heappop(funcs)
            gys = [output.grad for output in f.outputs]
            gxs = as_tuple(f.backward(*gys))
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    if x.creator not in seen_set:
                        seen_set.add(x.creator)
                        heapq.heappush(funcs, x.creator)


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = as_tuple(self.forward(*xs))
        outputs = [Variable(as_array(y)) for y in ys]
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
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

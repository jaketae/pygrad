import contextlib
import heapq
import weakref

import numpy as np


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    prev_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Cofig, name, prev_value)


def no_grad():
    return using_config("enable_backprop", False)


class Variable:
    def __init__(self, data):
        self.grad = None
        self.creator = None
        self.generation = 0
        self.data = np.asarray(data)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        return self

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        seen_set = set(funcs)
        while funcs:
            f = heapq.heappop(funcs)
            gys = [output().grad for output in f.outputs]
            gxs = as_tuple(f.backward(*gys))
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if not (x.creator is None or x.creator in seen_set):
                    seen_set.add(x.creator)
                    heapq.heappush(funcs, x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().clear_grad()


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = as_tuple(self.forward(*xs))
        outputs = [Variable(y).set_creator(self) for y in ys]
        if Config.enable_backprop:
            self.inputs = inputs
            self.generation = max([x.generation for x in inputs])
            self.outputs = [weakref.ref(output) for output in outputs]
        if len(outputs) > 1:
            return outputs
        return outputs[0]

    def __lt__(self, other):
        return self.generation > other.generation

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


def as_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x

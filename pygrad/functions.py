import numpy as np

from pygrad.core import Function


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, grad):
        return grad * np.exp(self.input.data)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, grad):
        return grad * 2 * self.input.data


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)

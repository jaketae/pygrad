import numpy as np

from pygrad.core import Function


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return gy * np.exp(self.inputs[0])


def exp(x):
    return Exp()(x)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return 2 * gy * self.inputs[0]


def square(x):
    return Square()(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        return gy * np.cos(self.inputs[0])


def sin(x):
    return Sin()(x)

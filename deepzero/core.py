import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None


class Function:
    def __call__(self, x):
        self.x = x
        y = self.forward(x.data)
        return Variable(y)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, grad):
        return grad * np.exp(self.x.data)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, grad):
        return grad * 2 * self.x.data


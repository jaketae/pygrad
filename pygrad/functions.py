import numpy as np

from pygrad.core import Function, as_variable
from pygrad.utils import _sum_to, handle_shape


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
        return gy * cos(self.inputs[0])


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        return -gy * sin(self.inputs[0])


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y * y)


def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy):
        return reshape(gy, self.x_shape)


@handle_shape
def reshape(x, shape):
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes):
        self.axes = axes

    def forward(self, x):
        return x.transpose(self.axes)

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        inv_axes = np.argsort(self.axes)
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy):
        return sum_to(gy, self.x_shape)


@handle_shape
def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return _sum_to(x, self.shape)

    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)


@handle_shape
def sum_to(x, shape):
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)

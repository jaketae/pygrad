import numpy as np

from pygrad.core import Function, as_variable
from pygrad.utils import _log_sum_exp, _sum_to, handle_shape


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return gy * exp(self.inputs[0])


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


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x1 - x0
        return np.average(diff ** 2)

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x1 - x0
        gx0 = 2 * gy * diff / len(diff)
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None
        if b.data is not None:
            gb = sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class Sigmoid(Function):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y * (1 - y)


def sigmoid(x):
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        y = x.copy()
        y[y < 0] = 0
        return y

    def backward(self, gy):
        x = self.inputs[0]
        mask = x.data > 0
        return gy * mask


def relu(x):
    return ReLU()(x)


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x < 0] *= self.slope
        return y

    def backward(self, gy):
        x = self.inputs[0]
        mask = x.data > 0
        mask[mask <= 0] = self.slope
        return gy * mask


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)


class GetItem(Function):
    def __init__(self, slice):
        self.slice = slice

    def forward(self, x):
        return x[self.slice]

    def backward(self, gy):
        x = self.inputs[0]
        gx = np.zeros_like(x.data)
        np.add.at(gx, self.slice, gy.data)
        return gx


def get_item(x, slice):
    return GetItem(slice)(x)


class Softmax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        y = np.exp(x - x.max(axis=self.axis, keepdims=True))
        return y / y.sum(axis=self.axis, keepdims=True)

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        gx_sum = gx.sum(axis=self.axis, keepdims=True)
        return gx - y * gx_sum


def softmax(x, axis=-1):
    return Softmax(axis)(x)


class Log(Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, gy):
        x = self.inputs[0]
        return gy / x


def log(x):
    return Log()(x)


class LogSoftMax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        log_z = _log_sum_exp(x, self.axis)
        return x - log_z

    def backward(self, gy):
        y = self.outputs[0]()
        return gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)


def log_softmax(x, axis=-1):
    return LogSoftMax(axis)(x)


class SoftmaxCrossEntropy(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x, t):
        if x.ndim != 2:
            raise ValueError(
                f"expected `x` predicted values to be 2D, "
                f"but received {x.ndim}D variable instead"
            )
        if t.ndim != 1:
            raise ValueError(
                f"expected `t` label to be 1D, "
                f"but received {t.ndim}D variable instead"
            )
        num_labels = len(t)
        num_samples = x.shape[0]
        if num_labels != num_samples:
            raise ValueError(
                f"expected `x` and `t` to be of same length, "
                f"but received {num_samples} samples, {num_labels} labels"
            )
        log_z = _log_sum_exp(x, self.axis)
        log_p = x - log_z
        log_p = log_p[:, t.ravel()]
        return -np.average(log_p)

    def backward(self, gy):
        x, t = self.inputs
        batch_size, num_class = x.shape
        y = softmax(x)
        t_onehot = np.eye(num_class)[t.data]
        return (y - t_onehot) * gy / batch_size


def softmax_cross_entropy(x, t, axis=-1):
    return SoftmaxCrossEntropy(axis)(x, t)


class Dropout(Function):
    def __init__(self, dropout, train):
        self.train = train
        self.dropout = dropout.data

    def forward(self, x):
        dropout = self.dropout
        if not (0 <= dropout <= 1):
            raise ValueError("`dropout` must be between 0 and 1")
        if self.train:
            self.mask = np.random.randn(*x.shape) > dropout
            self.scale = np.array(1 - dropout)
            return x * self.mask / self.scale
        return x

    def backward(self, gy):
        if self.train:
            return gy * self.mask / self.scale
        return gy


def dropout(x, dropout, train):
    return Dropout(dropout, train)(x)


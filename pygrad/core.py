import contextlib
import heapq
import warnings
import weakref

import numpy as np

import pygrad


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    prev_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, prev_value)


def no_grad():
    return using_config("enable_backprop", False)


def as_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Variable:

    __array_priority__ = 100

    def __init__(self, data, name=None):
        if isinstance(data, Variable):
            data = data.data
        self.data = np.asarray(data)
        data_dtype = self.data.dtype
        if not np.can_cast(data_dtype, np.number):
            raise TypeError(
                f"invalid data type '{data_dtype.type.__name__}' for `data`"
            )
        if name and not isinstance(name, str):
            raise TypeError(f"invalid data type '{type(name).__name__}' for `name`")
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        if isinstance(other, Variable):
            return np.array_equal(self.data, other.data)
        return False

    def __repr__(self):
        data_string = str(self.data).replace("\n", "\n" + " " * 9)
        if self.name is None:
            return f"Variable({data_string})"
        return f"Variable({data_string}), {self.name}"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        if self.ndim == 0:
            return self
        return pygrad.functions.transpose(self)

    def transpose(self, *axes):
        if self.ndim == 0:
            return self
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)) or axes[0] is None:
            axes = axes[0]
        if len(axes) > self.ndim:
            raise pygrad.exceptions.AxisError(axes, self.ndim)
        return pygrad.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        try:
            return pygrad.functions.sum(self, axis, keepdims)
        except np.AxisError:
            raise pygrad.exceptions.AxisError(axis, self.ndim)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape, (tuple, list)):
            shape = shape[0]
        return pygrad.functions.reshape(self, shape)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        return self

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.creator is None:
            raise RuntimeError("backward pass on a root variable")
        if self.data.size != 1:
            warnings.warn("attempting backward pass on a non-scalar variable")
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
        funcs = [self.creator]
        seen_set = set(funcs)
        while funcs:
            f = heapq.heappop(funcs)
            gys = [output().grad for output in f.outputs]
            with using_config("enable_backprop", create_graph):
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


class Parameter(Variable):
    pass


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = as_tuple(self.forward(*xs))
        outputs = [as_variable(y) for y in ys]
        if Config.enable_backprop:
            self.inputs = inputs
            self.generation = max([x.generation for x in inputs])
            self.outputs = [weakref.ref(output.set_creator(self)) for output in outputs]
        if len(outputs) > 1:
            return outputs
        return outputs[0]

    def __lt__(self, other):
        return self.generation > other.generation

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy
        gx1 = gy
        if x0.shape != x1.shape:
            gx0 = pygrad.functions.sum_to(gx0, x0.shape)
            gx1 = pygrad.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def add(x0, x1):
    return Add()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy
        gx1 = -gy
        if x0.shape != x1.shape:
            gx0 = pygrad.functions.sum_to(gx0, x0.shape)
            gx1 = pygrad.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def sub(x0, x1):
    return Sub()(x0, x1)


def rsub(x0, x1):
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = pygrad.functions.sum_to(gx0, x0.shape)
            gx1 = pygrad.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0, x1):
        if 0 in x1:
            raise ZeroDivisionError("division by variable containing zero")
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = -gy * x0 / (x1 ** 2)
        if x0.shape != x1.shape:
            gx0 = pygrad.functions.sum_to(gx0, x0.shape)
            gx1 = pygrad.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    return Div()(x0, x1)


def rdiv(x0, x1):
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0]
        return gy * self.c * x ** (self.c - 1)


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = pygrad.functions.get_item

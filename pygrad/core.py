from __future__ import annotations

import contextlib
import heapq
import warnings
import weakref
from typing import (
    Callable,
    ContextManager,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np

import pygrad


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value: bool) -> Iterator[None]:
    prev_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, prev_value)


def no_grad() -> ContextManager[None]:
    return using_config("enable_backprop", False)


def as_tuple(x) -> tuple:
    if not isinstance(x, tuple):
        return (x,)
    return x


def as_variable(x) -> Variable:
    if isinstance(x, Variable):
        return x
    return Variable(x)


class Variable:

    __array_priority__ = 100

    def __init__(self, data, name: str = None):
        if isinstance(data, Variable):
            data: np.ndarray = data.data  # type: ignore
        self.data = np.asarray(data)
        data_dtype = self.data.dtype
        if not np.can_cast(data_dtype, np.number):
            raise TypeError(
                f"invalid data type '{data_dtype.type.__name__}' for `data`"
            )
        if name and not isinstance(name, str):
            raise TypeError(f"invalid data type '{type(name).__name__}' for `name`")
        self.name = name
        self.grad: Optional[Variable] = None
        self.creator: Optional[Function] = None
        self.generation: int = 0

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Variable):
            return np.array_equal(self.data, other.data)
        return False

    def __repr__(self) -> str:
        data_string = str(self.data).replace("\n", "\n" + " " * 9)
        if self.name is None:
            return f"Variable({data_string})"
        return f"Variable({data_string}), {self.name}"

    def __neg__(self) -> Variable:
        return Neg()(self)

    def __add__(self, other) -> Variable:
        return Add()(self, other)

    def __radd__(self, other) -> Variable:
        return Add()(self, other)

    def __sub__(self, other) -> Variable:
        return Sub()(self, other)

    def __rsub__(self, other) -> Variable:
        return Sub()(other, self)

    def __mul__(self, other) -> Variable:
        return Mul()(self, other)

    def __rmul__(self, other) -> Variable:
        return Mul()(self, other)

    def __truediv__(self, other) -> Variable:
        return Div()(self, other)

    def __rtruediv__(self, other) -> Variable:
        return Div()(other, self)

    def __pow__(self, other) -> Variable:
        return Pow(other)(self)

    def __getitem__(self, key: Union[int, slice]) -> Variable:
        return pygrad.functions.get_item(self, key)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> Variable:
        if self.ndim == 0:
            return self
        return pygrad.functions.transpose(self)

    def transpose(self, *axes: int) -> Variable:
        if self.ndim == 0:
            return self
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)) or axes[0] is None:
            axes = axes[0]
        if len(axes) > self.ndim:
            raise pygrad.exceptions.AxisError(axes, self.ndim)
        return pygrad.functions.transpose(self, None or axes)

    def sum(self, axis: int = None, keepdims: bool = False) -> Variable:
        try:
            return pygrad.functions.sum(self, axis, keepdims)
        except np.AxisError:
            raise pygrad.exceptions.AxisError(axis, self.ndim)

    @overload
    def reshape(self, *shape: int) -> Variable:
        ...

    @overload
    def reshape(self, shape: Sequence[int]) -> Variable:
        ...

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return pygrad.functions.reshape(self, shape)

    def set_creator(self, func: Function) -> Variable:
        self.creator = func
        self.generation = func.generation + 1
        return self

    def clear_grad(self) -> None:
        self.grad = None

    def backward(self, retain_grad: bool = False, create_graph: bool = False) -> None:
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
        variables = [as_variable(x) for x in inputs]
        xs = [x.data for x in variables]
        ys = as_tuple(self.forward(*xs))
        outputs = [as_variable(y) for y in ys]
        if Config.enable_backprop:
            self.inputs = variables
            self.generation = max([x.generation for x in variables])
            self.outputs = [weakref.ref(output.set_creator(self)) for output in outputs]
        if len(outputs) > 1:
            return outputs
        return outputs[0]

    def __lt__(self, other: Function) -> bool:
        return self.generation > other.generation

    forward: Callable[..., np.ndarray]

    def backward(
        self, gys: Variable
    ) -> Union[Variable, Tuple[Optional[Variable], ...]]:
        raise NotImplementedError


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = gy
        gx1 = gy
        if x0.shape != x1.shape:
            gx0 = pygrad.functions.sum_to(gx0, x0.shape)
            gx1 = pygrad.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: Variable) -> Variable:
        return -gy


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 - x1

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = gy
        gx1 = -gy
        if x0.shape != x1.shape:
            gx0 = pygrad.functions.sum_to(gx0, x0.shape)
            gx1 = pygrad.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 * x1

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = pygrad.functions.sum_to(gx0, x0.shape)
            gx1 = pygrad.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        if 0 in x1:
            raise ZeroDivisionError("division by variable containing zero")
        return x0 / x1

    def backward(self, gy: Variable) -> Tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = -gy * x0 / (x1 ** 2)
        if x0.shape != x1.shape:
            gx0 = pygrad.functions.sum_to(gx0, x0.shape)
            gx1 = pygrad.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** self.c

    def backward(self, gy: Variable) -> Variable:
        x = self.inputs[0]
        return gy * self.c * x ** (self.c - 1)

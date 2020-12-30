# Autograd

Automatic differentiation, also referred to as automatic gradient computation or autograd, is at the heart of PyGrad's design. PyGrad computes gradient values by building a computational graph, following a define-by-run paradigm that maximizes ease of usability. 

## Variable Class

PyGrad adds a layer of abstraction on top of NumPy's `ndarray` class. For the most part, `Variable` acts much like a `ndarray` object.

```python
>>> from pygrad import Variable
>>> a = Variable(5)
>>> b = Variable(3)
>>> a + b
Variable(8)
```

PyGrad can, of course, also deal with arrays and tensors.

```python
>>> m1 = Variable([[1, 2], [3, 4]])
>>> m2 = Variable([[1, 1], [1, 1]])
>>> m1 + m2
Variable([[2 3]
          [4 5]])
```

Since PyGrad uses NumPy as its backend, it supports broadcasting as well as other matrix operations including transpose, reshape, and matrix multiplication.

```python
>>> m3 = a + m1
>>> m3
Variable([[4 5]
          [6 7]])
>>> m3.reshape(1, 4)
Variable([[4 5 6 7]])
```

## Function Class

PyGrad constructs computation graphs by creating a reference between variables and functions. Specifically, each PyGrad `Function` object has pointers to both its input and output `Variable` instances. `Variable` instances, in turn, contain reference to its `creator`, which is a `Function` object. Below is a simple example that demonstrates this relationship.

In this example, we demonstrate this relationship via the `Square` class.

```python
>>> from pygrad import functions as F
>>> x = Variable(2)
>>> square = F.Square()
>>> y = square(x)
```

Given this setup, we can now access the function that created `y`, which, as expected, is a `Square` instance.

```python
>>> y.creator
<pygrad.functions.Square object at 0x7fa63e090fd0>
```

The `square` function also has pointers to both its input and output `Variables`. 

```python
>>> square.inputs
[Variable(2)]
>>> square.outputs
[<weakref at 0x7fa63ea26650; to 'Variable' at 0x7fa63ab18350>]
```

In this case, the `square` function only has one input and output; for other non-unary functions, `Function.inputs` will return a list with two or more `Variable`s.

## Backpropagation

Every `Variable` instance has `data` and `grad` attributes. `data` stores the value of the variable itself as a `ndarray`, whereas `grad` stores the values of its gradients. When first initialized, all `Variable` instances have a `grad` value of `None`.

```python
>>> x = Variable(1)
>>> x.data
array(1)
>>> print(x.grad)
None
```

However, when PyGrad functions are applied on the `Variable` instance, it now belongs to a portion of a newly constructed computation graph. This means that PyGrad is ready to perform backpropagation, thus making `grad` take an actual value. To obtain gradients, simply call the `backward()` method on a leaf variable.

```python
>>> x = Variable(3)
>>> y = x * x
>>> y.backward()
>>> x.grad
Variable(6)
```

When a `backward()` method is called on a `Variable` instance, PyGrad traverses the computation graph to backpropagate throughout the entire chain. In doing so, it calls on the `backward()` method of each `Function` callable that are also part of the graph as `creator`s of some `Variable` objects. 

## Memory Optimization

PyGrad implements a few optimizations to make computation more efficient. 

### Weak References

For memory efficiency and garbage collection purposes, PyGrad stores `weakref` objects instead of the reference themselves. This eliminates circular references, thus allowing Python to garbage collect more efficiently. 

### Non-retained Gradients

By default, PyGrad will erase the gradient values of any intermediate `Variable` object in the middle of the computation graph. 

```python
>>> x = Variable(3)
>>> y = x * x
>>> z = y + y
>>> z.backward()
>>> print(y.grad)
None
```

This behavior is desirable since most computations only require the gradient of some parameter of interest. Removing the gradients of intermediary variables can save memory. However, it is possible to suppress this behavior by explicitly setting `retain_grad=True` in the `backward()` call.

```python
>>> z.backward(retain_grad=True)
>>> z.grad
Variable(1)
>>> y.grad
Variable(2)
>>> x.grad
Variable(12)
```


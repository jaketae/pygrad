import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward_recursive(self):
        creator_func = self.creator
        if creator_func is not None:
            x = creator_func.input
            x.grad = creator_func.backward(self.grad)
            x.backward()

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        creators = [self.creator]
        while creators:
            creator_func = creators.pop()
            x, y = creator_func.input, creator_func.output
            x.grad = creator_func.backward(y.grad)

            if x.creator is not None:
                creators.append(x.creator)


class Function:
    def __call__(self, input_):
        y = self.forward(input_.data)
        output = Variable(y)
        output.set_creator(self)
        self.input = input_
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


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


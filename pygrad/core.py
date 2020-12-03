import numpy as np


class Variable:
    def __init__(self, data, creator=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")

        self.grad = None
        self.data = data
        self.creator = creator

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
        output = Variable(as_array(y), creator=self)
        self.input = input_
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


def as_array(x):
    return np.asarray(x) if np.isscalar(x) else x

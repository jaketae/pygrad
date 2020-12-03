from deepzero.core import Exp, Square


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


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)

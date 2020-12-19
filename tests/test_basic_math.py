import unittest

import numpy as np

from pygrad.core import Variable
from pygrad.functions import square
from pygrad.utils import numerical_grad


class TestSquare(unittest.TestCase):
    def setUp(self):
        self.x = Variable(np.random.rand(1))
        self.y = square(self.x)

    def test_forward(self):
        expected = self.x.data ** 2
        self.assertEqual(self.y.data, expected)

    def test_backward(self):
        self.y.backward()
        expected = self.x.data * 2
        self.assertEqual(self.x.grad, expected)

    def test_gradient_check(self):
        self.y.backward()
        expected = numerical_grad(square, self.x)
        self.assertTrue(np.allclose(self.x.grad, expected))

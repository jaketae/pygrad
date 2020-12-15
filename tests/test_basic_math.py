import unittest

import numpy as np

from pygrad.core import Variable
from pygrad.functions import square


class TestSquare(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

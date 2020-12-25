from pygrad.utils import set_module


@set_module("pygrad")
class AxisError(IndexError, ValueError):
    def __init__(self, axis, ndim):
        super(AxisError, self).__init__(
            f"axis {axis} is out of bounds for variable of dimension {ndim}"
        )

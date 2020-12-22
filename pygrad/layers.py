from pygrad.core import Parameter


class Layer:
    def __init__(self):
        self.params = set()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.params.add(name)
        super(Layer, self).__setattr__(name, value)

from pygrad.layers import Layer


class Model(Layer):
    def num_params(self):
        return sum([param.size for param in self.params()])

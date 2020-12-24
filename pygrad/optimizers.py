class Optimizer:
    def __init__(self, target):
        self.hooks = []
        self.target = target

    def step(self):
        params = [p for p in self.target.params() if p.grad]
        for hook in self.hooks:
            hook(params)
        for param in params:
            self.step_one(param)

    def step_one(self, param):
        raise NotImplementedError

    def add_hook(self, hook):
        self.hooks.append(hook)


class SGD(Optimizer):
    def __init__(self, target, lr=1e-2):
        super(SGD, self).__init__(target)
        self.lr = lr

    def step_one(self):
        param.data -= self.lr * param.grad.data


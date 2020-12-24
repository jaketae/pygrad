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
    def __init__(self, target, lr=1e-2, momentum=0.9):
        super(SGD, self).__init__(target)
        self.vs = {}
        self.lr = lr
        self.momentum = momentum

    def step_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        v = self.momentum * self.vs[v_key]
        v -= self.lr * param.grad.data
        param.data += v
        self.vs[v_key] = v


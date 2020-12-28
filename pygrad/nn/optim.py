import numpy as np


def _check_lr(lr):
    if 0 > lr:
        raise ValueError(f"expected learning rate to be larger than 0, but got {lr}")


class Optimizer:
    def __init__(self, params):
        self.hooks = []
        self.params = tuple(params)

    def step(self):
        for hook in self.hooks:
            hook(params)
        for param in self.params:
            self.step_one(param)

    def step_one(self, param):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.clear_grad()

    def add_hook(self, hook):
        self.hooks.append(hook)


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.9):
        _check_lr(lr)
        super(SGD, self).__init__(params)
        self.vs = {}
        self.lr = lr
        self.momentum = momentum

    def step_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)
        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10):
        _check_lr(lr)
        super(AdaGrad, self).__init__(params)
        self.gs = {}
        self.lr = lr
        self.eps = eps

    def step_one(self, param):
        g_key = id(param)
        if g_key not in self.gs:
            self.gs[g_key] = np.zeros_like(param.data)
        grad = param.grad.data
        g = self.gs[g_key]
        g += grad * grad
        param.data -= self.lr * grad / (np.sqrt(g + self.eps))


class AdaDelta(Optimizer):
    def __init__(self, params, rho=0.95, eps=1e-6):
        super(AdaDelta, self).__init__(params)
        self.gs = {}
        self.dxhs = {}
        self.rho = rho
        self.eps = eps

    def step_one(self, param):
        key = id(param)
        if key not in self.gs:
            self.gs[key] = np.zeros_like(param.data)
            self.dxhs[key] = np.zeros_like(param.data)
        rho = self.rho
        eps = self.eps
        grad = param.grad.data
        g = self.gs[key]
        g *= rho
        g += (1 - rho) * grad * grad
        dxh = self.dxhs[key]
        dx = np.sqrt((dxh + eps) / (g + eps)) * grad
        dxh *= rho
        dxh += (1 - rho) * dx * dx
        param.data -= dx

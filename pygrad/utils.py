def numerical_grad(f, x, eps=1e-4):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

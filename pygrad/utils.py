def numerical_grad(f, x, eps=1e-4):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    if lead < 0:
        raise ValueError("`shape` has more dimensions than provided variable")
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

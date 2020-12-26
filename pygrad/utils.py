import os
import subprocess
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pygrad.core import Parameter


def numerical_grad(f, x, eps=1e-4):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def handle_shape(func):
    def wrapper(x, shape):
        if x.shape == shape:
            return x
        return func(x, shape)

    return wrapper


def set_module(module):
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func

    return decorator


def _sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    if lead < 0:
        raise ValueError("`shape` has more dimensions than provided variable")
    lead_axis = tuple(range(lead))
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    if y.shape != shape:
        raise RuntimeError(f"variable cannot be summed to shape {shape}")
    return y


def _log_sum_exp(x, axis):
    m = x.max(axis=axis, keepdims=True)
    y = np.exp(x - m).sum(axis=axis, keepdims=True)
    return np.add(m, np.log(y))


def write_dot_graph(model, dpi):
    nodes = {}
    edges = {}
    funcs = [output().creator for output in model.outputs]
    seen_set = set(funcs)
    while funcs:
        f = funcs.pop()
        node, edge = make_dot(f)
        nodes.update(node)
        edges.update(edge)
        for input_ in f.inputs:
            creator = input_.creator
            if creator and creator not in seen_set:
                seen_set.add(creator)
                funcs.append(creator)
    txt = ""
    for node_id, properties in nodes.items():
        txt += (
            f'{node_id} [label="{properties[0]} | {{ in | out }} | '
            f'{{ {properties[1]} | {properties[2]} }}"]\n'
        )
    for root, child in edges.items():
        txt += f"{root} -> {child}\n"
        if root not in nodes:
            txt += f'{root} [label="Variable"]\n'
    return (
        f'digraph g \n{{edge [fontname = "helvetica"];'
        f'graph [fontname = "helvetica", dpi={dpi}];'
        f'node [fontname = "helvetica", shape=Mrecord];\n{txt}}}'
    )


def make_dot(f):
    f_inputs = [input_ for input_ in f.inputs if not isinstance(input_, Parameter)]
    input_shapes = [input_.shape for input_ in f_inputs]
    output_shapes = [output().shape for output in f.outputs]
    name = f.__class__.__name__
    id_ = id(f)
    node = {id_: [name, input_shapes, output_shapes]}
    creator_ids = [id(input_.creator) for input_ in f_inputs]
    edge = {creator_id: id_ for creator_id in creator_ids}
    return node, edge


def _check_graphviz():
    try:
        subprocess.run("dot -V", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise ImportError(
            "please install graphviz (https://graphviz.gitlab.io/download/)"
        )


def plot_model(model, to_file="graph.png", dpi=300):
    _check_graphviz()
    graph = write_dot_graph(model, dpi)
    tmp_dir = os.path.join(os.path.expanduser("~"), ".pygrad")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_file = os.path.join(tmp_dir, "graph.dot")
    with open(graph_file, "w+") as f:
        f.write(graph)
    extension = os.path.splitext(to_file)[-1][1:]
    save_path = os.path.join(os.getcwd(), to_file)
    cmd = f"dot {graph_file} -T {extension} -o {save_path}"
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        raise FileNotFoundError(f"no such file or directory {save_path}")
    plt.axis("off")
    try:
        from IPython import display

        display.set_matplotlib_formats("retina")
        return diplay.Image(filename=to_file)
    except NameError:
        import matplotlib.image as mpimg

        img = mpimg.imread(to_file)
        plt.imshow(img)
        plt.show()

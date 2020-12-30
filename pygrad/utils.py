import os
import subprocess
from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pygrad.core import Function, Parameter, Variable


def numerical_grad(f: Function, x, eps: float = 1e-4) -> Variable:
    return (f(x + eps) - f(x - eps)) / (2 * eps)


ShapeFuncType = Callable[[Any, Tuple[int, ...]], Variable]


def handle_shape(func: ShapeFuncType) -> ShapeFuncType:
    def wrapper(x, shape):
        if x.shape == shape:
            return x
        return func(x, shape)

    return wrapper


def set_module(module: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        if module is not None:
            func.__module__ = module
        return func

    return decorator


def _sum_to(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
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


def _log_sum_exp(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    y = np.exp(x - m).sum(axis=axis, keepdims=True)
    return np.add(m, np.log(y))


def write_dot_graph(model, dpi: int) -> str:
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


def make_dot(f: Function) -> Tuple[dict, dict]:
    f_inputs = [input_ for input_ in f.inputs if not isinstance(input_, Parameter)]
    input_shapes = [input_.shape for input_ in f_inputs]
    output_shapes = [output().shape for output in f.outputs]
    name = f.__class__.__name__
    id_ = id(f)
    node = {id_: [name, input_shapes, output_shapes]}
    creator_ids = [id(input_.creator) for input_ in f_inputs]
    edge = {creator_id: id_ for creator_id in creator_ids}
    return node, edge


def _check_graphviz() -> None:
    try:
        subprocess.run("dot -V", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise ImportError(
            "please install graphviz (https://graphviz.gitlab.io/download/)"
        )


def plot_model(model, to_file: str = "graph.png", dpi: int = 300) -> None:
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
        from IPython import display  # noqa: F401

        return diplay.Image(filename=to_file, retina=True)  # type: ignore
    except NameError:
        import matplotlib.image as mpimg

        img = mpimg.imread(to_file)
        plt.imshow(img)
        plt.show()

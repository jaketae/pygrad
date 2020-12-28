# PyGrad

PyGrad is a NumPy-based pure Python mini automatic differentiation library. It provides a clear, minimal way to construct computational graphs, calculate gradients, and solve optimization problems, following a define-by-run paradigm. Building on top of these strengths as an autogrid library, PyGrad offers a minimal neural network API, inspired most by [PyTorch](https://pytorch.org). 

## Installation

PyGrad is available on PyPi. 

```
pip install pygrad
```

To build and develop from source, clone this repository via

```
git clone https://github.com/jaketae/pygrad.git
```

## Credits

PyGrad is heavily based upon [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3/tree/master/dezero), an educational library introduced in the book [Deep Learning from Scratch 3](https://koki0702.github.io/dezero-book/en/index.html). Much of the initial code base was adapted from DeZero. PyGrad's neural network API was largely inspired by PyTorch. [Chainer](https://chainer.org) should also be mentioned in this list as well, as PyGrad's precursor DeZero itself adapted many features from Chainer's original code base. Last but not least, PyGrad would not have been made possible without NumPy. Our acknowledgement goes to all the developers who put their time and effort into developing these wonderful libraries. 

## License

Released under the [MIT License](LICENSE.md).
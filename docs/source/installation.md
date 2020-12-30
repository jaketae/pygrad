# Get Started

## Python Version

The minimum requirement for running PyGrad is Python 3.6. For optimum performance, we recommend using Python version 3.7 or higher. 

## Installation

PyGrad can be installed through `pip`.

```
pip install pygrad
```

To use the latest development version of PyGrad, run

```
pip install https://github.com/jaketae/pygrad/archive/master.zip
```

Verify your installation in the current environment via

```
pip list | grep pygrad
```

## Optional Dependencies

To generate visual summaries of neural network architectures, PyGrad uses Graphviz, a graph visualization package based on the DOT language. Graphviz is not required to use PyGrad itself, but is required to use PyGrad's model plotting feature.

To install Graphviz on Linux, run

```
sudo apt install graphviz
```

On macOS, 

```
brew install graphviz
```

On Windows, Graphviz can be installed through Windows Package Manager:

```
winget install graphviz
```

For more detailed instructions, consult the [Graphviz download page](https://graphviz.org/download/).
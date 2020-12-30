# Contributing

Thank you for taking the time to contribute to PyGrad. We welcome any and all forms of contributions, including code patches, document improvements, bug reports, and feature requests. Below are some guidelines for reference when contributing to the code base.

## Environment Setup

Begin by forking this repository. Then, clone the forked repository to your local machine via

```
git clone https://github.com/your-github-username/pygrad.git
cd pygrad
```

Create a new Python virtual environment. PyGrad's recommended environment requires Python 3.7 or later.

```
python -m venv pygrad-dev
```

Activate the environment according to the protocol of our local operating system.

```
pygrad-dev\Scripts\activate.bat # Windows
source pygrad-dev/bin/activate # Unix or macOS
```

Install necessary dependencies specified in `requirements.txt`.

```
pip install -r requirements.txt
```

Finally, add an upstream remote via

```
git remote add upstream https://github.com/jaketae/pygrad.git
```

To verify that the upstream has correctly been added to the list of remotes, run `git remove -v`.

## Code Checks

PyGrad's coding style is dictated by [black](https://black.readthedocs.io/en/stable/). Additionally, [isort](https://pycqa.github.io/isort/) is used to sort module imports. For [PEP 8](https://www.python.org/dev/peps/pep-0008/) errors and static type checking, we use [flake8](https://flake8.pycqa.org/en/latest/) and [mypy](https://mypy.readthedocs.io), respectively.

Minimal modifications have been made to the default configurations of each module to prevent collisions. For more information on these modified configurations, please refer to [`setup.cfg`](https://github.com/jaketae/pygrad/blob/master/setup.cfg).

For convenience purposes, we have created a [`Makefile`](https://github.com/jaketae/pygrad/blob/master/Makefile) that contains commands to check and lint files in relevant directories. To lint, run the following command in the root directory:

```
make style
```

To perform quality checks on the code base, type

```
make quality
```

Note that the quality check will not run the linter; it will only show errors and traces.

Last but not least, to run all unit tests, run

```
make test
```

Please make sure to run all three commands before opening a pull request.

## Pull Request

Before making a pull request, rebase your local branch. To perform a rebase, first fetch upstream changes.

```
git fetch upstream
```

Then, rebase with the upstream branch via

```
git rebase upstream/master
```

Resolve merge conflicts, if any, and continue the rebase. When rebasing is complete, open a pull request on the PyGrad repository.

## Final Words

Thanks again for your interest in this project. We appreciate your invaluable input!

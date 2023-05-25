# Simulating spikes with data-driven control

This code implements a data-driven controller that adapts to potentially changing
dynamical systems. The goal is to relate it to biological neural circuits.

## Table of Contents

- [Installation](#installation)
- [Example usage](#example-usage)
- [Questions?](#questions)

## Installation

It is strongly recommended to use a virtual environment when working with this code. The
instructions below use `conda` environments, but a similar approach can be taken with
`venv` or `virtualenv`.

If you do not yet have `conda` installed, the easiest way to get started is with
[Miniconda](https://docs.conda.io/en/latest/miniconda.html). Follow the installation
instructions for your system.

Next, create a new environment and install the necessary pre-requisites using

```sh
conda env create -f environment.yml
```

Finally, activate the environment and install the `ddc` package:

```sh
conda activate ddc
pip install -e .
```

The `-e` marks this as an "editable" install — this means that changes made to the code
will automatically take effect without having to reinstall the package. This is mainly
useful for developers. If you just want to use the code in the package, you can omit the
`-e`.

## Example usage

The scripts in the [`sandbox`](sandbox) folder test various parts of the code. They are
best thought of as Jupyter notebooks in script format. You can either run them
cell-by-cell using [VSCode's](https://code.visualstudio.com/)
[interactive mode](https://code.visualstudio.com/docs/python/jupyter-support-py), or you
can convert them to bona fide notebooks using
[Jupytext](https://github.com/mwouts/jupytext) and run in, e.g., Jupyter.

## Questions?

Come talk to me! (Or open an issue on GitHub.)

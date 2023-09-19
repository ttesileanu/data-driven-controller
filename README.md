# Simulating spikes with data-driven control

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-red.svg)](https://www.python.org/downloads/release/python-390/)

This code implements a data-driven controller that adapts to potentially changing
dynamical systems. The goal is to relate it to biological neural circuits. The main
inspiration is the data-enabled predictive control (DeePC) method from Coulson, J.,
Lygeros, J., & Dörfler, F. (2019). *Data-Enabled Predictive Control: In the Shallows of
the DeePC*.

The code has two parts: a package called [`ddc`](src/ddc) implementing DeePC as well as
some classes that help with simulating dynamical systems; and a [sandbox](sandbox)
folder containing various examples and experiments. The package is automatically
installed when using the instructions below. See the [example usage](#example-usage)
section for navigating the examples.

## Table of Contents

- [Installation](#installation)
- [Example usage](#example-usage)
- [Questions?](#questions)

## Installation

It is strongly recommended to use a virtual environment when working with this code.
Installation using `conda` and `pip` is supported.

### Installing with conda

If you do not yet have `conda` installed, the easiest way to get started is with
[Miniconda](https://docs.conda.io/en/latest/miniconda.html). Follow the installation
instructions for your system.

Next, create a new environment and install the `ddc` package together with the necessary
pre-requisites by running the following in a terminal:

```sh
conda env create -f environment.yml
```

This creates a `conda` environment called `ddc` that can be activated using

```sh
conda activate ddc
```

The installation makes an editable install of `ddc`—this means that changes made to the
code automatically take effect without having to reinstall the package.

### Installing with pip

This method requires that you have a proper Python install on your system. Note that,
while most modern operating systems come with some version of Python pre-installed, this
is meant to be used for OS-related tasks and in most cases it is a very bad idea to use
the system-installed Python for user purposes!

To install a non-system Python, some options are outlined in
[The Hitchhiker's Guide to Python](https://docs.python-guide.org/starting/installation/#installation-guides),
although many options exist. If you do not want to deal with this, it's best to use
conda instead.

Once you have a proper Python install, you can create a new virtual environment by
running
```sh
python -m venv env
```

This creates a subfolder of the current folder called `env` containing the files for the
virtual environment. Next we need to activate the environment and install the package:

```sh
source env/bin/activate
pip install -e .
```

As above, this makes an editable install of `ddc` so that changes you make to the code
automatically take effect.

## Example usage

The scripts in the [`sandbox`](sandbox) folder test various parts of the code. They are
best thought of as Jupyter notebooks in script format. You can either run them
cell-by-cell using [VSCode's](https://code.visualstudio.com/)
[interactive mode](https://code.visualstudio.com/docs/python/jupyter-support-py), or you
can convert them to bona fide notebooks using
[Jupytext](https://github.com/mwouts/jupytext). You can then run those notebooks in,
e.g., Jupyter.

## Background and design

The project started with an ad-hoc implementation based on the behavioral approach to
dynamical system modeling. Some form of these attempts can still be found in the (now
obsolete) [`DDController`](src/ddc/dd_controller.py).

Eventually we found about DeePC, which is fundamentally the same as our original idea,
but a bit cleaner. We implemented [`DeepControl`](src/ddc/deepc.py) following DeePC,
removing a few bits that seemed unnecessary (like the slack on the observed `y` and `u`)
and adding some bits (like the ability to perform affine control, following an idea
from Berberich, J., Köhler, J., Müller, M. A., & Allgöwer, F. (2022). *Linear Tracking
MPC for Nonlinear Systems — Part II: The Data-Driven Case.* IEEE Transactions on
Automatic Control, 67(9), 4406–4421).

All the while we kept an eye on our goal of finding a biologically plausible
implementation of these techniques. This meant focusing on a receding-horizon
implementation that can be thought of as "online", and keeping the algorithms we use
simple (e.g., avoiding the use of a convex optimizer and instead sticking to simple
linear algebra).

### Simulation code

As mentioned above, the code contains some components that help simulate dynamical
systems. The [`GeneralSystem`](src/ddc/general_system.py) class helps implement a
generic, potentially non-linear, discrete-time dynamical system with additive state and
observation Gaussian noise. The generation of the noise is simplified by the
[`GaussianDistribution`](src/ddc/gauss.py) class.

The more specialized [`AffineControlSystem`](src/ddc/general_system.py) builds upon
`GeneralSystem` to support only affine control (i.e., where the time evolution is a sum
of a purely state-dependent and a purely control-dependent part).

Finally, the [`LinearSystem`](src/ddc/linear_system.py) implements linear dynamical
systems, allowing the user to simply specify the relevant system matrices.

All of these have a common interface for generating samples, based on the `run()`
method. This method can run an arbitrary number of steps, can accept a "control
schedule" specifying the system inputs during those steps, and can do batch runs, where
multiple trajectories are simulated at once. See the
[`GeneralSystem.run()` docstring](src/ddc/general_system.py) for details.

### Control code

The most up-to-date control code is provided by the [`DeepControl`](src/ddc/deepc.py)
class. This supports inputs and outputs of arbitrary dimension, it supports partial
observations through the use of non-trivial lag vectors (`ini_length` option), it
allows for control targets that are different from zero (`target` option), it uses
L2 regularization for the optimization stage, is capable of running in either online
(receding horizon) or offline modes, can add noise to its output to avoid losing
persistency of excitation in online mode, plus a few more features. See the
[`DeepControl` docstring](src/ddc/deepc.py) for details.

Check out some of th examples in the [sandbox](sandbox) folder to see how the
`DeepControl` class is used. The only thing that is not very intuitive about it is that
you need to have a "seed" interval during which random control is applied to the system
and its output is observed before the class can start making control suggestions.

The now-obsolete [`DDController`](src/ddc/dd_controller.py) class implements a similar
method to DeePC that we had developed before we found DeePC. This code is no longer used
in any of the examples, but can be found in older commits.

The constrained optimizations used by `DDController` and `DeepControl` are supported by
the code in [`solvers.py`](src/ddc/solvers.py).

## Extending the code

The most important part of the controller is the [`DeepControl.plan()`](src/ddc/deepc.py)
function, which uses the recent history of inputs and outputs to suggest a control plan
that should bring the system closer to the target. This is the place where most changes
to the model would make sense.

For implementing switching DeePC, one could also change `DeepControl.feed()` to update
the Hankels appropriately.

## Questions?

If you run into any trouble, please open an issue on GitHub.

# %% [markdown]
## Linear system

# %%
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

from typing import Optional

# %%
from scipy.linalg import sqrtm


class GaussianDistribution:
    def __init__(self, cov: torch.Tensor, mu: Optional[torch.Tensor] = None):
        """Generate samples from a multivariate normal distribution.

        :param mu: mean of the distribution; default: zero
        :param cov: covariance matrix of distribution
        """
        self.cov = cov

        assert self.cov.ndim == 2
        self.n = self.cov.shape[0]
        assert self.cov.shape[1] == self.n

        if mu is None:
            self.mu = torch.zeros(self.n)
        else:
            self.mu = mu
            assert self.mu.ndim == 1
            assert len(self.mu) == self.n

        self.trafo = None
        self._prepare()

    def sample(
        self, count: Optional[int] = None, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Generate samples from the multivariate Gaussian.

        :param count: number of samples to generate; if not provided, generate only one
            sample; note that the dimensionality of the output is different depending on
            whether count is provided or not (see below)
        :param generator: random number generator to use; default: PyTorch default
        :return: random sample(s); if `count` is not provided, this has shape `(n,)`,
            where `n` is the dimensionality of the Guassian; if `count` is provided, the
            shape is `(count, n)` -- note that this is different from the former case
            even if `count == 1`
        """
        if count is None:
            single = True
            count = 1
        else:
            single = False

        y = torch.randn(count, self.n, generator=generator)
        x = y @ self.trafo

        x = x + self.mu

        if single:
            assert x.shape[0] == 1
            x = x[0]

        return x

    def _prepare(self):
        """Precalculate the transformation matrix from standard-normal to the requested
        distribution.
        """
        self.trafo = torch.tensor(sqrtm(self.cov))


# %%
class LinearSystem:
    def __init__(
        self,
        evolution: torch.Tensor,
        control: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
        state_noise: Optional[torch.Tensor] = None,
        observation_noise: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """Create a linear dynamical system.

        The system obeys

            x[t + 1] = evolution @ x[t] + control @ u[t] + state_noise ,
            y[t] = observation @ x[t] + observation_noise ,

        where `u[t]` is the control signal and `y[t]` are the observations. Note that
        the state and observation noise are assumed independent for now.

        Note that the state can be a single column vector (the default), or a matrix. In
        the latter case each column of the state vector evolves independently -- this is
        the batch case.

        :param evolution: evolution matrix
        :param control: control matrix
        :param observation: observation matrix; default: identity
        :param state_noise: covariance matrix for state noise (assumed white); default:
            the zero matrix, i.e., no state noise
        :param observation_noise: covariance matrix for observation noise (assumed
            white); default: the zero matrix, i.e., no observation noise
        :param initial_state: value of system state at initial time, `x[0]`; default:
            zero vector
        :param generator: random number generator for state and observation noise;
            default: use PyTorch's default generator
        """
        self.evolution = evolution.detach().clone()
        self.control = control.detach().clone()

        self.state_dim = self.evolution.shape[0]
        assert self.evolution.shape[1] == self.state_dim

        self.control_dim = self.control.shape[1]
        assert self.control.shape[0] == self.state_dim

        if observation is None:
            self.observation = torch.eye(self.state_dim)
        else:
            self.observation = observation.detach().clone()
        self.observation_dim = self.observation.shape[0]
        assert self.observation.shape[1] == self.state_dim

        if state_noise is None:
            self.state_noise = torch.zeros((self.state_dim, self.state_dim))
            self.has_state_noise = False
        else:
            self.state_noise = state_noise.detach().clone()
            self._state_gaussian = GaussianDistribution(self.state_noise)
            self.has_state_noise = True
        assert self.state_noise.shape[0] == self.state_dim
        assert self.state_noise.shape[1] == self.state_dim

        if observation_noise is None:
            self.observation_noise = torch.zeros(
                (self.observation_dim, self.observation_dim)
            )
            self.has_observation_noise = False
        else:
            self.observation_noise = observation_noise.detach().clone()
            self._observation_gaussian = GaussianDistribution(self.observation_noise)
            self.has_observation_noise = True
        assert self.observation_noise.shape[0] == self.observation_dim
        assert self.observation_noise.shape[1] == self.observation_dim

        if initial_state is None:
            self.state = torch.zeros((self.state_dim, 1))
            self.batch_size = 1
        else:
            if initial_state.ndim == 1:
                initial_state = initial_state[:, None]
            self.state = initial_state.detach().clone()
            self.batch_size = self.state.shape[1]
        assert self.state.shape[0] == self.state_dim

        self.generator = generator

    def run(
        self, n_steps: Optional[int] = None, control_plan: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run the dynamical system for a number of steps.

        If the number of steps is provided without a `control_plan`, the function will
        simulate an autonomous system (i.e., with control set to 0).

        If a `control_plan` is provided, it will set the control inputs received by the
        system. Generally the number of steps can be inferred from the shape of the
        `control_plan` (see below); if `n_steps` is also provided, it should be
        compatible with the shape of the `control_plan`.

        :param n_step: number of steps to simulate; if not provided, infer from
            `control_plan` (see above)
        :param control_plan: control inputs to be provided to the system; should have
            shape `(n, control_dim)` or `(n, control_dim, batch_size); in the former
            case, if `batch_size > 1`, the same control is used for all samples
        :return: the system observations, with shape `(n, observation_dim, batch_size)`,
            where `n` is the number of steps
        """
        if control_plan is not None:
            assert control_plan.ndim in [2, 3]
            if n_steps is not None:
                assert n_steps == control_plan.shape[0]
            else:
                n_steps = control_plan.shape[0]

            if control_plan.ndim < 3:
                control_plan = torch.tile(control_plan[..., None], (self.batch_size,))
            assert control_plan.ndim == 3
            assert control_plan.shape[-1] == self.batch_size

            assert control_plan.shape[1] == self.control_dim

        observations = torch.empty(n_steps, self.observation_dim, self.batch_size)
        for i in range(n_steps):
            # observe
            y = self.observation @ self.state
            if self.has_observation_noise:
                y += self._observation_gaussian.sample(
                    self.batch_size, generator=self.generator
                ).T
            observations[i] = y

            # evolve
            x = self.evolution @ self.state
            if control_plan is not None:
                x += self.control @ control_plan[i]
            if self.has_state_noise:
                x += self._state_gaussian.sample(
                    self.batch_size, generator=self.generator
                ).T

            self.state = x

        return observations


# %% [markdown]
### Some tests
#### One-dimensional, no noise

# %%
model = LinearSystem(
    evolution=torch.Tensor([[0.9]]),
    control=torch.Tensor([[1.0]]),
    initial_state=torch.Tensor([1.0]),
)
observations = model.run(100)

with dv.FigureManager() as (_, ax):
    ax.plot(observations.squeeze())
    ax.set_xlabel("time")
    ax.set_ylabel("observation = state")

# %% [markdown]
#### One-dimensional, observation noise, multiple batches

# %%
model = LinearSystem(
    evolution=torch.Tensor([[0.9]]),
    control=torch.Tensor([[1.0]]),
    observation_noise=0.005 * torch.eye(1),
    initial_state=torch.ones(1, 200),
)
observations = model.run(100)

with dv.FigureManager() as (_, ax):
    ax.plot(observations.squeeze(), c="k", lw=1.0, alpha=0.04)
    ax.set_xlabel("time")
    ax.set_ylabel("observation = state")

# %% [markdown]
#### One-dimensional, state noise, multiple batches

# %%
model = LinearSystem(
    evolution=torch.Tensor([[0.9]]),
    control=torch.Tensor([[1.0]]),
    state_noise=0.005 * torch.eye(1),
    initial_state=torch.ones(1, 200),
)
observations = model.run(100)

with dv.FigureManager() as (_, ax):
    ax.plot(observations.squeeze(), c="k", lw=1.0, alpha=0.04)
    ax.set_xlabel("time")
    ax.set_ylabel("observation = state")

# %% [markdown]
#### One-dimensional, with control

# %%
model = LinearSystem(
    evolution=torch.Tensor([[0.9]]),
    control=torch.Tensor([[1.0]]),
    initial_state=torch.tensor([1.0]),
)
control_plan = torch.hstack(
    (
        0.1 * torch.ones(20),
        torch.zeros(20),
        0.2 * torch.ones(20),
        torch.zeros(40),
    )
)[:, None]
observations = model.run(control_plan=control_plan)

with dv.FigureManager() as (_, ax):
    ax.plot(observations.squeeze())
    ax.plot(control_plan.squeeze(), "k", lw=1.0, alpha=0.5, label="control")
    ax.set_xlabel("time")
    ax.set_ylabel("observation = state")
    ax.legend(frameon=False)

# %% [markdown]
#### One-dimensional, with control and observation noise

# %%
model = LinearSystem(
    evolution=torch.Tensor([[0.9]]),
    control=torch.Tensor([[1.0]]),
    observation_noise=0.005 * torch.eye(1),
    initial_state=torch.ones(1, 200),
)
control_plan = torch.hstack(
    (
        0.1 * torch.ones(20),
        torch.zeros(20),
        0.2 * torch.ones(20),
        torch.zeros(40),
    )
)[:, None]
observations = model.run(control_plan=control_plan)

with dv.FigureManager() as (_, ax):
    ax.plot(observations.squeeze(), c="C0", lw=1.0, alpha=0.04)
    ax.plot(control_plan.squeeze(), "k", lw=1.0, alpha=0.5, label="control")
    ax.set_xlabel("time")
    ax.set_ylabel("observation = state")
    ax.legend(frameon=False)

# %% [markdown]
#### One-dimensional, with control and state noise

# %%
model = LinearSystem(
    evolution=torch.Tensor([[0.9]]),
    control=torch.Tensor([[1.0]]),
    state_noise=0.005 * torch.eye(1),
    initial_state=torch.ones(1, 200),
)
control_plan = torch.hstack(
    (
        0.1 * torch.ones(20),
        torch.zeros(20),
        0.2 * torch.ones(20),
        torch.zeros(40),
    )
)[:, None]
observations = model.run(control_plan=control_plan)

with dv.FigureManager() as (_, ax):
    ax.plot(observations.squeeze(), c="C0", lw=1.0, alpha=0.04)
    ax.plot(control_plan.squeeze(), "k", lw=1.0, alpha=0.5, label="control")
    ax.set_xlabel("time")
    ax.set_ylabel("observation = state")
    ax.legend(frameon=False)

# %% [markdown]
#### One-dimensional, with per-sample control

# %%
model = LinearSystem(
    evolution=torch.Tensor([[0.9]]),
    control=torch.Tensor([[1.0]]),
    initial_state=torch.ones(1, 3),
)
control_plan = torch.hstack(
    (
        torch.vstack(
            (
                0.1 * torch.ones(30, 1),
                0.0 * torch.ones(70, 1),
            )
        ),
        torch.vstack(
            (
                0.0 * torch.ones(40, 1),
                0.1 * torch.ones(60, 1),
            )
        ),
        torch.vstack(
            (
                0.0 * torch.ones(24, 1),
                0.1 * torch.ones(24, 1),
                0.0 * torch.ones(24, 1),
                0.1 * torch.ones(28, 1),
            )
        ),
    )
)[:, None]
observations = model.run(control_plan=control_plan)

with dv.FigureManager() as (_, ax):
    for i in range(control_plan.shape[-1]):
        ax.plot(observations[:, :, i].squeeze(), c=f"C{i}")
        ax.plot(
            control_plan[:, :, i].squeeze(),
            "--",
            c=f"C{i}",
            lw=1.0,
            alpha=0.5,
            label="control",
        )
    ax.set_xlabel("time")
    ax.set_ylabel("observation = state")
    ax.legend(frameon=False)

# %% [markdown]
#### Two-dimensional
rot_mat = torch.FloatTensor([[np.cos(0.2), np.sin(0.2)], [-np.sin(0.2), np.cos(0.2)]])
model = LinearSystem(
    evolution=0.99 * rot_mat,
    control=torch.eye(2),
    initial_state=torch.tensor([1.0, 0.0]),
)
observations = model.run(25)

with dv.FigureManager(figsize=(4, 4)) as (_, ax):
    theta = np.linspace(0, 2 * np.pi, 50)
    circle = [np.cos(theta), np.sin(theta)]
    for rad in [0.5, 1.0]:
        ax.plot(rad * circle[0], rad * circle[1], "k:", alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, observations.shape[0]))
    for i in range(observations.shape[0]):
        y = observations[i].squeeze()
        ax.plot([0, y[0]], [0, y[1]], c=colors[i], lw=1.0)
    h = ax.scatter(
        observations[:, 0, 0],
        observations[:, 1, 0],
        c=np.arange(observations.shape[0]),
    )
    dv.colorbar(h, ax=ax)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect(1)

# %% [markdown]
#### Two-dimensional with observation noise
rot_mat = torch.FloatTensor([[np.cos(0.2), np.sin(0.2)], [-np.sin(0.2), np.cos(0.2)]])
model = LinearSystem(
    evolution=0.99 * rot_mat,
    control=torch.eye(2),
    observation_noise=0.001 * torch.tensor([[0.55, -0.45], [-0.45, 0.55]]),
    initial_state=torch.tile(torch.tensor([[1.0], [0.0]]), (1, 25)),
)
observations = model.run(25)

with dv.FigureManager(figsize=(4, 4)) as (_, ax):
    theta = np.linspace(0, 2 * np.pi, 50)
    circle = [np.cos(theta), np.sin(theta)]
    for rad in [0.5, 1.0]:
        ax.plot(rad * circle[0], rad * circle[1], "k:", alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, observations.shape[0]))
    for i in range(observations.shape[0]):
        ys = observations[i].squeeze()
        for k in range(ys.shape[1]):
            y = ys[:, k]
            ax.plot([0, y[0]], [0, y[1]], c=colors[i], lw=1.0, alpha=0.2)
    for k in range(observations.shape[-1]):
        h = ax.scatter(
            observations[:, 0, k],
            observations[:, 1, k],
            c=np.arange(observations.shape[0]),
            s=5,
            alpha=0.3,
        )
        dv.colorbar(h, ax=ax)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect(1)


# %% [markdown]
#### Two-dimensional with state noise
rot_mat = torch.FloatTensor([[np.cos(0.2), np.sin(0.2)], [-np.sin(0.2), np.cos(0.2)]])
model = LinearSystem(
    evolution=0.99 * rot_mat,
    control=torch.eye(2),
    state_noise=0.0001 * torch.tensor([[0.55, -0.45], [-0.45, 0.55]]),
    initial_state=torch.tile(torch.tensor([[1.0], [0.0]]), (1, 25)),
)
observations = model.run(25)

with dv.FigureManager(figsize=(4, 4)) as (_, ax):
    theta = np.linspace(0, 2 * np.pi, 50)
    circle = [np.cos(theta), np.sin(theta)]
    for rad in [0.5, 1.0]:
        ax.plot(rad * circle[0], rad * circle[1], "k:", alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, observations.shape[0]))
    for i in range(observations.shape[0]):
        ys = observations[i].squeeze()
        for k in range(ys.shape[1]):
            y = ys[:, k]
            ax.plot([0, y[0]], [0, y[1]], c=colors[i], lw=1.0, alpha=0.2)
    for k in range(observations.shape[-1]):
        h = ax.scatter(
            observations[:, 0, k],
            observations[:, 1, k],
            c=np.arange(observations.shape[0]),
            s=5,
            alpha=0.3,
        )
        dv.colorbar(h, ax=ax)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect(1)

# %% [markdown]
#### Two-dimensional with control
rot_mat = torch.FloatTensor([[np.cos(0.2), np.sin(0.2)], [-np.sin(0.2), np.cos(0.2)]])
model = LinearSystem(
    evolution=0.99 * rot_mat,
    control=torch.tensor([[1.0], [0.3]]),
    initial_state=torch.tensor([1.0, 0.0]),
)
# control_plan = torch.empty(n, control_dim, batch_size)
control_plan = torch.cat(
    (
        torch.zeros((3, 1, 1)),
        0.1 * torch.ones((2, 1, 1)),
        torch.zeros(10, 1, 1),
        0.1 * torch.ones((2, 1, 1)),
        torch.zeros((10, 1, 1)),
    )
)
observations = model.run(control_plan=control_plan)

with dv.FigureManager(figsize=(4, 4)) as (_, ax):
    theta = np.linspace(0, 2 * np.pi, 50)
    circle = [np.cos(theta), np.sin(theta)]
    for rad in [0.5, 1.0]:
        ax.plot(rad * circle[0], rad * circle[1], "k:", alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, observations.shape[0]))
    for i in range(observations.shape[0]):
        y = observations[i].squeeze()
        ax.plot([0, y[0]], [0, y[1]], c=colors[i], lw=1.0)
    h = ax.scatter(
        observations[:, 0, 0],
        observations[:, 1, 0],
        c=np.arange(observations.shape[0]),
    )
    dv.colorbar(h, ax=ax)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect(1)

# %%

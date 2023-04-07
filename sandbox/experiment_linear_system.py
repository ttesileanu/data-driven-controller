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

# %% [markdown]
# ## A data-driven controller


# %%
class DDController:
    def __init__(
        self,
        observation_dim: int,
        control_dim: int,
        history_length: int,
        seed_length: int = 1,
        control_horizon: int = 1,
        observation_match_cost: float = 1.0,
        control_match_cost: float = 1.0,
        target_cost: float = 0.01,
        control_cost: float = 0.01,
        control_sparsity: float = 0.0,
    ):
        """Data-drive controller.

        This acts by leveraging the behavioral approach to dynamical systems: a system
        is identified by the fact that its possible trajectories are restricted to a
        certain subspace. Observing actual trajectories can then be used to identify
        this subspace and, thus, the system.

        Given a matrix `Y` in which each column is a sequence of observations of a
        dynamical system and a matrix `U` in which each column is a sequence of control
        inputs to the system, and assuming that the number of columns is large enough,
        any new sequence of controls `u` and observations `y` can be written as a linear
        combination of the earlier data:

            z = Z alpha ,

        where `Z = vstack((Y, U))`, `z = vstack((y, u))`, and `alpha` is a vector of
        coefficients. The matrices `Y` and `U` can be obtained from a single long
        sequence in the form of Hankel matrices -- and that is what we will be doing
        here.

        In our case, we want to establish a control plan that maximally reduces the
        magnitude of the observations after a certain horizon, using the earlier
        measurements to predict the behavior of the system. To perform this prediction
        we need to start with a "seed" where the observations and control are known --
        the seed should be sufficiently long to fully specify the (unobserved) internal
        state of the system.

        More specifically, suppose the system's state is fully identified by observing
        `seed_length` steps and suppose that we're trying to minimize the system's
        output a number `control_horizon` of steps in the future. Since we want to make
        sure that the full (unobserved) state of the system is under control, we also
        evaluate the target output by summing over `seed_length` steps. The number of
        prior samples used in the prediction is given by `history_length`.

        The control planning is set up as an optimization problem where the coefficients
        `alpha` are inferred such that a loss function containing five terms is
        minimized:

            L = (observation_match_cost * (predicted - measured seed observations)) ** 2
              + (control_match_cost * (predicted - measured seed controls)) ** 2
              + (target_cost * (predicted observations at horizon)) ** 2
              + (control_cost * (predicted control after seed)) ** 2
              + control_sparsity * L1_norm(predicted control after seed) .

        The inferred coefficients are then used to predict the optimal control. Note
        that apart from the L1 regularization term, this is just a least-squares
        problem.

        :param observation_dim: dimensionality of measurements
        :param control_dim: dimensionality of control
        :param history_length: number of columns to use in the Hankel matrices
        :param seed_length: number of measurements needed to fully specify an internal
            state
        :param control_horizon: how far ahead to aim for reduced measurements
        :param observation_match_cost: multiplier for matching observation values
        :param control_match_cost: multiplier for matching seed control values
        :param target_cost: multiplier for minimizing observation values at horizon
        :param control_cost: multiplier for minimizing overall control magnitude
        :param control_sparsity: amount of sparsity-inducing, L1 regularizer; NOT YET
            IMPLEMENTED
        """
        self.observation_dim = observation_dim
        self.control_dim = control_dim
        self.history_length = history_length
        self.seed_length = seed_length
        self.control_horizon = control_horizon
        self.observation_match_cost = observation_match_cost
        self.control_match_cost = control_match_cost
        self.target_cost = target_cost
        self.control_cost = control_cost
        self.control_sparsity = control_sparsity
        if self.control_sparsity != 0:
            assert NotImplementedError("control_sparsity not implemented yet")

        self.observation_history = None
        self.control_history = None

    def feed(self, observation: torch.Tensor, control: torch.Tensor):
        """Register another measurement.

        At most `self.history_length` samples are kept.

        :param observation: output from the model
        :param control: control input
        """
        assert len(observation) == self.observation_dim
        assert len(control) == self.control_dim

        if self.observation_history is None:
            self.observation_history = observation.clone()
        else:
            self.observation_history = torch.vstack(
                (self.observation_history, observation)
            )
        if self.control_history is None:
            self.control_history = control.clone()
        else:
            self.control_history = torch.vstack((self.control_history, control))

        n = self.history_length + self.control_horizon + self.seed_length
        self.observation_history = self.observation_history[-n:]
        self.control_history = self.control_history[-n:]

    def plan(self) -> torch.Tensor:
        """Estimate optimal control plan given the current history.

        :return: control plan, shape `(control_horizon, control_dim)`
        """
        p = self.seed_length
        l = self.control_horizon
        n = self.history_length

        d = self.observation_dim
        c = self.control_dim

        if len(self.observation_history) < n + p + l:
            # not enough data yet
            return torch.zeros((l, c))

        obs = self.observation_history
        ctrl = self.control_history

        end = len(obs)
        assert len(ctrl) == end

        Y = torch.empty((2 * p * d, n))
        U = torch.empty(((p + l) * c, n))
        # target y and target z are zero
        y = torch.zeros((2 * p * d, 1))
        u = torch.zeros((p + l) * c, 1)
        for i in range(p):
            yi = i * d
            Y[yi : yi + d] = obs[i : i + n].T
            y[yi : yi + d] = obs[end + i - p][:, None]

            yi = (i + p) * d
            Y[yi : yi + d] = obs[l + i : l + i + n].T

            ui = i * c
            U[ui : ui + c] = ctrl[i : i + n].T
            u[ui : ui + c] = ctrl[end + i - p][:, None]

        for i in range(p, p + l):
            ui = i * c
            U[ui : ui + c] = ctrl[i : i + n].T

        Z = torch.vstack((Y, U))
        z = torch.vstack((y, u))

        # weigh using the appropriate coefficients
        Z_weighted = Z.clone()
        Z_weighted[: p * d] *= self.observation_match_cost
        Z_weighted[p * d : 2 * p * d] *= self.target_cost
        Z_weighted[2 * p * d : p * (2 * d + c)] *= self.control_match_cost
        Z_weighted[-l * c :] *= self.control_cost

        z_weighted = z.clone()
        z_weighted[: p * d] *= self.observation_match_cost
        # nothing to multiply for [p * d : 2 * p * d] as target y is zero

        z_weighted[2 * p * d : p * (2 * d + c)] *= self.control_match_cost
        # nothing to multiply for [-l * c :] as target u is zero

        # now solve z = Z alpha
        result = torch.linalg.lstsq(Z_weighted, z_weighted)
        z_hat = Z @ result.solution

        return z_hat[-l * c :].reshape((l, c))


# %% [markdown]
# ### Try the DD controller
# #### A 1d example

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
)
history_length = 25
control_horizon = 5
controller = DDController(
    1,
    1,
    history_length,
    control_horizon=control_horizon,
    control_cost=0,
    target_cost=0.1,
)

n_steps = 100
control = torch.tensor([0.0])
control_noise = 0.001
outputs = []
controls = []
for k in range(n_steps):
    # need noise for exploration
    control = control + control_noise * torch.randn(control.shape)
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(y, control)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = (
    controller.history_length + controller.control_horizon + controller.seed_length
)

outputs = torch.stack(outputs)
controls = torch.stack(controls)

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    yl = (outputs.min(), outputs.max())
    axs[0].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[0].plot(outputs.squeeze())
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("observation")
    axs[0].legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[1].plot(controls.squeeze())
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    axs[1].legend(frameon=False, loc="lower right")

# %% [markdown]
# #### A 2d example

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1, 0.0], [0.0, 0.8]]),
    control=torch.tensor([[1.0], [-0.3]]),
    initial_state=torch.tensor([[1.0], [0.5]]),
)
history_length = 25
control_horizon = 5
controller = DDController(
    2,
    1,
    history_length,
    control_horizon=control_horizon,
    control_cost=0,
    target_cost=0.1,
)

n_steps = 200
control = torch.tensor([0.0])
control_noise = 0.001
outputs = []
controls = []
for k in range(n_steps):
    # need noise for exploration
    control = control + control_noise * torch.randn(control.shape)
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(y, control)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = (
    controller.history_length + controller.control_horizon + controller.seed_length
)

outputs = torch.stack(outputs)
controls = torch.stack(controls)

# %%
with dv.FigureManager(3, 1, figsize=(6, 6)) as (_, axs):
    for i in range(2):
        ax = axs[i]
        output = outputs[:, i]
        yl = (output.min(), output.max())
        ax.fill_betweenx(
            yl,
            [0, 0],
            2 * [control_start],
            color="gray",
            alpha=0.5,
            edgecolor="none",
            label="no control",
        )
        ax.plot(output.squeeze())
        ax.set_xlabel("time")
        ax.set_ylabel(f"observation $y_{{{i + 1}}}$")
        ax.legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[2].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[2].plot(controls.squeeze())
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("control")
    axs[2].legend(frameon=False, loc="lower right")

# %% [markdown]
# #### 2d with rotation and observation noise

# %%
torch.manual_seed(42)
rot_mat = torch.FloatTensor([[np.cos(0.1), np.sin(0.1)], [-np.sin(0.1), np.cos(0.1)]])
scale_mat = torch.FloatTensor([[1.1, 0.0], [0.0, 0.8]])
transfer_mat = scale_mat @ rot_mat
model = LinearSystem(
    evolution=transfer_mat,
    control=torch.tensor([[1.0], [-0.3]]),
    observation_noise=torch.tensor([[0.1, 0.0], [0.0, 0.2]]),
    initial_state=torch.tensor([[1.0], [0.5]]),
)
history_length = 25
control_horizon = 5
controller = DDController(
    2,
    1,
    history_length,
    control_horizon=control_horizon,
    control_cost=0,
    target_cost=0.1,
)

n_steps = 200
control = torch.tensor([0.0])
control_noise = 0.001
outputs = []
controls = []
for k in range(n_steps):
    # need noise for exploration
    control = control + control_noise * torch.randn(control.shape)
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(y, control)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = (
    controller.history_length + controller.control_horizon + controller.seed_length
)

outputs = torch.stack(outputs)
controls = torch.stack(controls)

# %%
with dv.FigureManager(3, 1, figsize=(6, 6)) as (_, axs):
    for i in range(2):
        ax = axs[i]
        output = outputs[:, i]
        yl = (output.min(), output.max())
        ax.fill_betweenx(
            yl,
            [0, 0],
            2 * [control_start],
            color="gray",
            alpha=0.5,
            edgecolor="none",
            label="no control",
        )
        ax.plot(output.squeeze())
        ax.set_xlabel("time")
        ax.set_ylabel(f"observation $y_{{{i + 1}}}$")
        ax.legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[2].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[2].plot(controls.squeeze())
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("control")
    axs[2].legend(frameon=False, loc="lower right")

# %%
with dv.FigureManager() as (_, ax):
    h = dv.color_plot(
        outputs[:, 0].numpy(),
        outputs[:, 1].numpy(),
        np.arange(len(outputs)),
        cmap="viridis",
        ax=ax,
    )
    cb = dv.colorbar(h)
    cb.set_label("time")
    ax.autoscale()
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

# %% [markdown]
# #### 2d with rotation and state noise

# %%
torch.manual_seed(42)
rot_mat = torch.FloatTensor([[np.cos(0.1), np.sin(0.1)], [-np.sin(0.1), np.cos(0.1)]])
scale_mat = torch.FloatTensor([[1.1, 0.0], [0.0, 0.8]])
transfer_mat = scale_mat @ rot_mat
model = LinearSystem(
    evolution=transfer_mat,
    control=torch.tensor([[1.0], [-0.3]]),
    state_noise=torch.tensor([[0.05, 0.0], [0.0, 0.1]]),
    initial_state=torch.tensor([[1.0], [0.5]]),
)
history_length = 25
control_horizon = 5
controller = DDController(
    2,
    1,
    history_length,
    control_horizon=control_horizon,
    control_cost=0,
    target_cost=0.1,
)

n_steps = 200
control = torch.tensor([0.0])
control_noise = 0.001
outputs = []
controls = []
for k in range(n_steps):
    # need noise for exploration
    control = control + control_noise * torch.randn(control.shape)
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(y, control)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = (
    controller.history_length + controller.control_horizon + controller.seed_length
)

outputs = torch.stack(outputs)
controls = torch.stack(controls)

# %%
with dv.FigureManager(3, 1, figsize=(6, 6)) as (_, axs):
    for i in range(2):
        ax = axs[i]
        output = outputs[:, i]
        yl = (output.min(), output.max())
        ax.fill_betweenx(
            yl,
            [0, 0],
            2 * [control_start],
            color="gray",
            alpha=0.5,
            edgecolor="none",
            label="no control",
        )
        ax.plot(output.squeeze())
        ax.set_xlabel("time")
        ax.set_ylabel(f"observation $y_{{{i + 1}}}$")
        ax.legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[2].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[2].plot(controls.squeeze())
    axs[2].set_xlabel("time")
    axs[2].set_ylabel("control")
    axs[2].legend(frameon=False, loc="lower right")

# %%
with dv.FigureManager() as (_, ax):
    h = dv.color_plot(
        outputs[:, 0].numpy(),
        outputs[:, 1].numpy(),
        np.arange(len(outputs)),
        cmap="viridis",
        ax=ax,
    )
    cb = dv.colorbar(h)
    cb.set_label("time")
    ax.autoscale()
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)


# %% [markdown]
# #### 2d rotation, observation noise, state noise, and partial observation

# %%
torch.manual_seed(42)
rot_mat = torch.FloatTensor([[np.cos(0.1), np.sin(0.1)], [-np.sin(0.1), np.cos(0.1)]])
scale_mat = torch.FloatTensor([[1.1, 0.0], [0.0, 0.8]])
transfer_mat = scale_mat @ rot_mat
model = LinearSystem(
    evolution=transfer_mat,
    observation=torch.tensor([[1.0, 0.0]]),
    control=torch.tensor([[1.0], [-0.3]]),
    observation_noise=torch.tensor([[0.2]]),
    state_noise=torch.tensor([[0.05, 0.0], [0.0, 0.1]]),
    initial_state=torch.tensor([[1.0], [0.5]]),
)
history_length = 25
control_horizon = 8
controller = DDController(
    1,
    1,
    history_length,
    seed_length=2,
    control_horizon=control_horizon,
    control_cost=0,
    target_cost=0.1,
)

n_steps = 200
control = torch.tensor([0.0])
control_noise = 0.001
outputs = []
controls = []
for k in range(n_steps):
    # need noise for exploration
    control = control + control_noise * torch.randn(control.shape)
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(y, control)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = (
    controller.history_length + controller.control_horizon + controller.seed_length
)

outputs = torch.stack(outputs)
controls = torch.stack(controls)

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    ax = axs[0]
    output = outputs[:, 0]
    yl = (output.min(), output.max())
    ax.fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    ax.plot(output.squeeze())
    ax.set_xlabel("time")
    ax.set_ylabel(f"observation $y_{{{i + 1}}}$")
    ax.legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[1].plot(controls.squeeze())
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    axs[1].legend(frameon=False, loc="lower right")

# %% [markdown]
# #### 2d rotation, observation noise, state noise, and partial observation

# %%
torch.manual_seed(42)
rot_mat = torch.FloatTensor([[np.cos(0.2), np.sin(0.2)], [-np.sin(0.2), np.cos(0.2)]])
scale_mat = torch.FloatTensor([[1.1, 0.0], [0.0, 0.8]])
transfer_mat = scale_mat @ rot_mat
model = LinearSystem(
    evolution=transfer_mat,
    observation=torch.tensor([[1.0, 0.0]]),
    control=torch.tensor([[1.0], [-0.3]]),
    observation_noise=torch.tensor([[0.2]]),
    state_noise=torch.tensor([[0.05, 0.0], [0.0, 0.1]]),
    initial_state=torch.tensor([[1.0], [0.5]]),
)
history_length = 25
control_horizon = 8
controller = DDController(
    1,
    1,
    history_length,
    seed_length=2,
    control_horizon=control_horizon,
    control_cost=0,
    target_cost=0.1,
)

n_steps = 2500
control = torch.tensor([0.0])
control_noise = 0.001
outputs = []
controls = []
for k in range(n_steps):
    # need noise for exploration
    control = control + control_noise * torch.randn(control.shape)
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(y, control)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = (
    controller.history_length + controller.control_horizon + controller.seed_length
)

outputs = torch.stack(outputs)
controls = torch.stack(controls)

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    ax = axs[0]
    output = outputs[:, 0]
    yl = (output.min(), output.max())
    ax.fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    ax.plot(output.squeeze())
    ax.set_xlabel("time")
    ax.set_ylabel(f"observation $y$")
    ax.legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[1].plot(controls.squeeze())
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    axs[1].legend(frameon=False, loc="lower right")

# %%

# %% [markdown]
# # Try to control a cart-pole system

# %%
from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# %%
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

from ddc import GeneralSystem, DeepControl

# %% [markdown]
# ## Define the simulator


class CartPole(GeneralSystem):
    def __init__(
        self,
        dt: float,
        M: float = 1.0,
        m: float = 0.1,
        l: float = 0.5,
        g: float = 9.8,
        observation=None,
        **kwargs,
    ):
        # parameters
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.dt = dt

        super().__init__(
            self.evolution_fct, observation=observation, state_dim=4, **kwargs
        )

    def evolution_fct(self, state: torch.Tensor, force: torch.Tensor):
        """Apply the given force to the cart and update the state by one time step."""
        # RK4
        h = self.dt
        k1 = self._gradient(state, force)
        k2 = self._gradient(state + (0.5 * h) * k1, force)
        k3 = self._gradient(state + (0.5 * h) * k2, force)
        k4 = self._gradient(state + h * k3, force)

        dstate = (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return state + dstate

    def _gradient(self, state: torch.Tensor, force: float) -> torch.Tensor:
        x, theta, dx, dtheta = state
        c = torch.cos(theta)
        s = torch.sin(theta)
        denom = self.M + self.m * s**2

        ml = self.m * self.l
        mg = self.m * self.g
        ddx = (force + ml * dtheta**2 * s - mg * s * c) / denom

        Mtotg = (self.M + self.m) * self.g
        ddtheta = (Mtotg * s - force * c - ml * dtheta**2 * s * c) / denom / self.l

        grad = torch.tensor([[dx], [dtheta], [ddx], [ddtheta]]).to(dtype=state.dtype)
        return grad


# %% [markdown]
# ### Test the simulation of the cart pole

# %%
pendulum = CartPole(dt=0.02, initial_state=torch.tensor([[0], [1e-6], [0], [0]]))

n_steps = 1000
phase = torch.zeros((4, n_steps))

for i in range(n_steps):
    phase[:, i] = pendulum.state.squeeze()
    pendulum.run(1)

# %%
with dv.FigureManager(2, 1, figsize=(4, 4), constrained_layout=True) as (_, axs):
    for i, ax in enumerate(axs):
        ax.plot(phase[i, :])
        ax.set_ylabel(["$x$", "$\\theta$"][i])

# %% [markdown]
# ## Long horizon, online

# %%
torch.manual_seed(42)
control_horizon = 12
controller = DeepControl(
    control_dim=1,
    ini_length=2,
    horizon=control_horizon,
    l2_regularization=1e-4,
    control_cost=1e-5,
    seed_noise_norm=2.0,
    affine=True,
    control_norm_clip=5.0,
)

model = CartPole(
    dt=0.02,
    M=100,
    observation=lambda state: state[[1, 2]],
    initial_state=torch.tensor([[0], [np.pi - 0.05], [0], [0]], dtype=torch.float64),
)

n_steps = 2000
planning_steps = 2
seed_length = 300

control_snippet = controller.generate_seed(seed_length, model.dtype)
control_start = len(control_snippet)
for k in range(n_steps // planning_steps):
    observation_snippet = model.run(control_plan=control_snippet)

    controller.feed(observation_snippet)
    control_plan = controller.plan()

    control_snippet = control_plan[:planning_steps]

observations = torch.stack(controller.history.observations)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

# %%
with dv.FigureManager(3, 1, figsize=(8, 6)) as (_, axs):
    axs[0].set_title("Online")

    for i, ax in enumerate(axs[:2]):
        crt_output = observations[:, i]

        yl = (min(crt_output.min(), 0), max(crt_output.max(), 0))
        ax.axhline(0, c="gray", ls=":", lw=1.0)
        ax.fill_betweenx(
            yl,
            [0, 0],
            2 * [control_start],
            color="gray",
            alpha=0.5,
            edgecolor="none",
            label="seed",
        )
        ax.plot(crt_output.squeeze(), lw=1.0)
        ax.set_xlabel("time")
        ax.set_ylabel(["$\\theta$", "$\\dot x$"][i])

    yl = (min(controls.min(), 0), max(controls.max(), 0))
    axs[-1].axhline(0, c="gray", ls=":", lw=1.0)
    axs[-1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="seed",
    )
    axs[-1].plot(controls.squeeze(), lw=1.0)
    axs[-1].set_xlabel("time")
    axs[-1].set_ylabel("control")


# %% [markdown]
# ## Long horizon, online, observe 1 - cosine

torch.manual_seed(42)
control_horizon = 12
controller = DeepControl(
    control_dim=1,
    ini_length=2,
    horizon=control_horizon,
    l2_regularization=1e-4,
    control_cost=1e-5,
    seed_noise_norm=2.0,
    affine=True,
    control_norm_clip=5.0,
)

model = CartPole(
    dt=0.02,
    M=100,
    observation=lambda state: state[[1, 2]],
    initial_state=torch.tensor([[0], [np.pi - 0.05], [0], [0]], dtype=torch.float64),
)

n_steps = 2000
planning_steps = 2
seed_length = 300

control_snippet = controller.generate_seed(seed_length, model.dtype)
control_start = len(control_snippet)
for k in range(n_steps // planning_steps):
    observation_snippet = model.run(control_plan=control_snippet)

    transformed_snippet = observation_snippet.clone()
    transformed_snippet[:, 0] = 1 - torch.cos(transformed_snippet[:, 0])
    transformed_snippet[:, 1] /= 100.0
    controller.feed(observation_snippet)
    control_plan = controller.plan()

    control_snippet = control_plan[:planning_steps]

observations = torch.stack(controller.history.observations)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

# %%
with dv.FigureManager(3, 1, figsize=(9, 5)) as (_, axs):
    axs[0].set_title("Online")

    for i, ax in enumerate(axs[:2]):
        crt_output = observations[:, i]

        yl = (min(crt_output.min(), 0), max(crt_output.max(), 0))
        ax.axhline(0, c="gray", ls=":", lw=1.0)
        ax.fill_betweenx(
            yl,
            [0, 0],
            2 * [control_start],
            color="gray",
            alpha=0.5,
            edgecolor="none",
            label="no control",
        )
        ax.plot(crt_output.squeeze(), lw=1.0)
        ax.set_xlabel("time")
        ax.set_ylabel(["$1 - \\cos(\\theta)$", "$\\dot x$"][i])

    yl = (min(controls.min(), 0), max(controls.max(), 0))
    axs[-1].axhline(0, c="gray", ls=":", lw=1.0)
    axs[-1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[-1].plot(controls.squeeze(), lw=1.0)
    axs[-1].set_xlabel("time")
    axs[-1].set_ylabel("control")

# %%

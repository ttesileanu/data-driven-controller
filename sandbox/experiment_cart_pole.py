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

from ddc import DDController

# %% [markdown]
# ## Define the simulator


class CartPole:
    def __init__(
        self,
        M: float = 1.0,
        m: float = 0.1,
        l: float = 0.5,
        g: float = 9.8,
        dt: float = 0.02,
    ):
        # parameters
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.dt = dt

        # current state: positions and velocities
        self.q = torch.zeros(2)  # [x, theta]
        self.dq = torch.zeros(2)

    def step(self, f: float):
        """Apply the given force to the cart and update the state by one time step."""
        theta = self.q[1]
        c = torch.cos(theta)
        s = torch.sin(theta)
        denom = self.M + self.m * s**2

        ml = self.m * self.l
        mg = self.m * self.g
        dtheta = self.dq[1]
        ddx = (f + ml * dtheta**2 * s - mg * s * c) / denom

        Mtotg = (self.M + self.m) * self.g
        ddtheta = (Mtotg * s - f * c - ml * dtheta**2 * s * c) / denom / self.l

        # ddq = torch.tensor([ddx, ddtheta]).to(dtype=self.q.dtype)

        # leap frog
        # change = (0.5 * self.dt) * ddq
        # self.dq += change
        # self.q += self.dt * self.dq
        # self.dq += change

        # Euler
        # change = self.dt * ddq
        # self.q += self.dt * self.dq
        # self.dq += change

        # RK4
        h = self.dt
        y = torch.hstack((self.q, self.dq))
        k1 = self._gradient(y, f)
        k2 = self._gradient(y + (0.5 * h) * k1, f)
        k3 = self._gradient(y + (0.5 * h) * k2, f)
        k4 = self._gradient(y + h * k3, f)

        dy = (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.q += dy[:2]
        self.dq += dy[2:]

    def _gradient(self, y: torch.Tensor, f: float) -> torch.Tensor:
        theta = y[1]
        c = torch.cos(theta)
        s = torch.sin(theta)
        denom = self.M + self.m * s**2

        ml = self.m * self.l
        mg = self.m * self.g
        dtheta = y[3]
        ddx = (f + ml * dtheta**2 * s - mg * s * c) / denom

        Mtotg = (self.M + self.m) * self.g
        ddtheta = (Mtotg * s - f * c - ml * dtheta**2 * s * c) / denom / self.l

        grad = torch.tensor([y[2], y[3], ddx, ddtheta]).to(dtype=y.dtype)
        return grad

    def astype(self, t: torch.dtype) -> "CartPole":
        self.q.to(dtype=t)
        self.dq.to(dtype=t)
        return self


# %% [markdown]
# ### Test the simulation of the cart pole

# %%
pendulum = CartPole()
pendulum.q[1] = 1e-6

n_steps = 1000
phase = torch.zeros((4, n_steps))

for i in range(n_steps):
    phase[:2, i] = pendulum.q
    phase[2:, i] = pendulum.dq
    pendulum.step(0.0)

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
controller = DDController(
    2,
    1,
    seed_length=2,
    averaging_factor=2.0,
    control_horizon=control_horizon,
    noise_handling="none",
    l2_regularization=5.0,
    noise_strength=0.01,
    output_cost=100.0,
    affine=True,
    # clip_control=(-40, 40),
    clip_control=(-1, 1),
)

pendulum = CartPole(M=100)

# initial state
pendulum.q[1] = np.pi - 0.05
# pendulum.q[1] = 1e-6

n_steps = 2000
for k in range(n_steps):
    # decide on control magnitude
    obs = torch.hstack((pendulum.q[[1]], pendulum.dq[[0]]))
    controller.feed(obs)
    control_plan = controller.plan()

    # run the model
    pendulum.step(control_plan[0].item())

control_start = controller.history_length

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

# %%
with dv.FigureManager(3, 1, figsize=(8, 6)) as (_, axs):
    axs[0].set_title("Online")

    for i, ax in enumerate(axs[:2]):
        crt_output = outputs[:, i]

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
        label="no control",
    )
    axs[-1].plot(controls.squeeze(), lw=1.0)
    axs[-1].set_xlabel("time")
    axs[-1].set_ylabel("control")


# %% [markdown]
# ## Long horizon, online, observe 1 - cosine

# %%
torch.manual_seed(42)
control_horizon = 20
controller = DDController(
    2,
    1,
    seed_length=2,
    averaging_factor=2.0,
    control_horizon=control_horizon,
    noise_handling="none",
    l2_regularization=5.0,
    noise_strength=0.01,
    output_cost=1000.0,
    affine=True,
    # clip_control=(-40, 40),
    clip_control=(-5, 5),
)

pendulum = CartPole()

# initial state
pendulum.q[1] = 1e-6

n_steps = 2000
for k in range(n_steps):
    # decide on control magnitude
    obs = torch.hstack((1 - torch.cos(pendulum.q[[1]]), pendulum.dq[[0]] / 100.0))
    controller.feed(obs)
    control_plan = controller.plan()

    # run the model
    pendulum.step(control_plan[0].item())

control_start = controller.history_length

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

# %%
with dv.FigureManager(3, 1, figsize=(9, 5)) as (_, axs):
    axs[0].set_title("Online")

    for i, ax in enumerate(axs[:2]):
        crt_output = outputs[:, i]

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

# %% [markdown]
# ## Long horizon, online, observe 1 - cosine, run more of the control plan at once

# %%
torch.manual_seed(42)
control_horizon = 24
controller = DDController(
    # 2,
    1,
    1,
    seed_length=2,
    averaging_factor=4.0,
    control_horizon=control_horizon,
    noise_handling="none",
    l2_regularization=5.0,
    noise_strength=0.01,
    output_cost=1000.0,
    affine=True,
    # clip_control=(-40, 40),
    clip_control=(-15, 15),
)

pendulum = CartPole()

# initial state
pendulum.q[1] = np.pi - 0.05

n_steps = 2000
ctrl_stride = 10
k = 0
while k < n_steps:
    if controller.observation_history is None or (
        len(controller.observation_history) < controller.history_length
    ):
        # decide on control magnitude
        # obs = torch.hstack((1 - torch.cos(pendulum.q[[1]]), pendulum.dq[[0]] / 100.0))
        obs = 1 - torch.cos(pendulum.q[[1]])
        controller.feed(obs)
        control_plan = controller.plan()

        # run the model
        pendulum.step(control_plan[0].item())
        k += 1
    else:
        for i in range(ctrl_stride):
            # obs = torch.hstack(
            #     (1 - torch.cos(pendulum.q[[1]]), pendulum.dq[[0]] / 100.0)
            # )
            obs = 1 - torch.cos(pendulum.q[[1]])
            controller.feed(obs)
            pendulum.step(control_plan[i].item())
            k += 1

        control_plan = controller.plan()

control_start = controller.history_length

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

# %%
with dv.FigureManager(2, 1, figsize=(9, 3)) as (_, axs):
    axs[0].set_title("Online")

    for i, ax in enumerate(axs[:1]):
        crt_output = outputs[:, i]

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

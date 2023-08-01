# %% [markdown]
# # Try to control a double inverted pendulum

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
# ## Define the simulators


class DoubleCartPole:
    def __init__(
        self,
        M: float = 2.0,
        m1: float = 1.0,
        m2: float = 0.3,
        l1: float = 0.5,
        l2: float = 0.4,
        g: float = 9.8,
        dt: float = 0.01,
    ):
        # parameters
        self.M = M
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.dt = dt

        # current state: generalized positions and velocities
        self.q = torch.zeros(3)  # [x, theta_1, theta_2]
        self.dq = torch.zeros(3)

    def step(self, f: float):
        """Apply the given force to the pendulum and update the state by one time step."""
        Mtot = self.M + self.m1 + self.m2
        m = self.m1 + self.m2

        theta1 = self.q[1]
        theta2 = self.q[2]
        c1 = torch.cos(theta1)
        s1 = torch.sin(theta1)
        c2 = torch.cos(theta2)
        s2 = torch.sin(theta2)

        dtheta = theta2 - theta1
        cdiff = torch.cos(dtheta)
        sdiff = torch.sin(dtheta)

        dq1sq = self.dq[1] ** 2
        dq2sq = self.dq[2] ** 2

        A = torch.tensor(
            [
                [Mtot, self.l1 * m * c1, self.m2 * self.l2 * c2],
                [m * c1, m * self.l1, self.m2 * self.l2 * cdiff],
                [c2, self.l1 * cdiff, self.l2],
            ]
        )
        b = torch.tensor(
            [
                [f + m * self.l1 * dq1sq * s1 + self.m2 * self.l2 * dq2sq * s2],
                [m * self.g * s1 + self.m2 * self.l2 * dq2sq * sdiff],
                [self.g * s2 - self.l1 * dq1sq * sdiff],
            ]
        )

        result = torch.linalg.lstsq(A, b)
        ddq = result.solution.squeeze()

        # leap frog
        change = (0.5 * self.dt) * ddq
        self.dq += change
        self.q += self.dt * self.dq
        self.dq += change


class DoubleInvertedPendulum:
    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 0.3,
        l1: float = 0.5,
        l2: float = 0.4,
        g: float = 9.8,
        dt: float = 0.01,
    ):
        # parameters
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.dt = dt

        # current state: generalized positions and velocities
        self.q = torch.zeros(2)  # [theta_1, theta_2]
        self.dq = torch.zeros(2)

    def step(self, torques: torch.Tensor):
        """Apply the given torques to the pendulums and update the state by one time
        step.
        """
        m = self.m1 + self.m2

        theta1 = self.q[0]
        theta2 = self.q[1]
        c1 = torch.cos(theta1)
        s1 = torch.sin(theta1)
        c2 = torch.cos(theta2)
        s2 = torch.sin(theta2)

        dtheta = theta2 - theta1
        cdiff = torch.cos(dtheta)
        sdiff = torch.sin(dtheta)

        dq1sq = self.dq[0] ** 2
        dq2sq = self.dq[1] ** 2

        A = torch.tensor(
            [
                [m * self.l1, self.m2 * self.l2 * cdiff],
                [self.l1 * cdiff, self.l2],
            ]
        )
        b = (
            torch.tensor(
                [
                    [m * self.g * s1 + self.m2 * self.l2 * dq2sq * sdiff],
                    [self.g * s2 - self.l1 * dq1sq * sdiff],
                ]
            )
            + torques
        )

        result = torch.linalg.lstsq(A, b)
        ddq = result.solution.squeeze()

        # leap frog
        change = (0.5 * self.dt) * ddq
        self.dq += change
        self.q += self.dt * self.dq
        self.dq += change


# %% [markdown]
# ### Test the simulation of double cart pole

# %%
pendulum = DoubleCartPole()
pendulum.q[1] = np.pi - 0.05
pendulum.q[2] = np.pi + 0.2

n_steps = 1000
phase = torch.zeros((6, n_steps))

for i in range(n_steps):
    phase[:3, i] = pendulum.q
    phase[3:, i] = pendulum.dq
    pendulum.step(0.0)

# %%
with dv.FigureManager(3, 1, figsize=(6, 4), constrained_layout=True) as (_, axs):
    for i, ax in enumerate(axs):
        ax.plot(phase[i, :])
        ax.set_ylabel(["$x$", "$\\theta_1$", "$\\theta_2$"][i])

# %% [markdown]
# ### Test the simulation of double inverted pendulum

# %%
pendulum = DoubleInvertedPendulum()
pendulum.q[0] = np.pi - 0.05
pendulum.q[1] = np.pi + 0.2

n_steps = 1000
phase = torch.zeros((4, n_steps))

for i in range(n_steps):
    phase[:2, i] = pendulum.q
    phase[2:, i] = pendulum.dq
    pendulum.step(torch.tensor([[0.0], [0.0]]))

# %%
with dv.FigureManager(2, 1, figsize=(6, 4), constrained_layout=True) as (_, axs):
    for i, ax in enumerate(axs):
        ax.plot(phase[i, :])
        ax.set_ylabel(["$\\theta_1$", "$\\theta_2$"][i])

# %% [markdown]
# ## Long horizon, online, double cart pole

# %%
torch.manual_seed(42)
control_horizon = 12
controller = DDController(
    2,
    1,
    seed_length=3,
    averaging_factor=4.0,
    control_horizon=control_horizon,
    noise_handling="none",
    l2_regularization=5.0,
    noise_strength=0.01,
    output_cost=1000.0,
    affine=True,
    clip_control=(-40, 40),
)

pendulum = DoubleCartPole(dt=0.01)

# initial state
pendulum.q[1] = np.pi - 0.05
pendulum.q[2] = np.pi + 0.2

n_steps = 2000
for k in range(n_steps):
    # decide on control magnitude
    controller.feed(pendulum.q[1:])
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
        ax.set_ylabel(["$\\theta_1$", "$\\theta_2$"][i])

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
# ## Long horizon, online, double inverted pendulum

# %%
torch.manual_seed(42)
control_horizon = 12
controller = DDController(
    2,
    1,
    seed_length=2,
    averaging_factor=4.0,
    control_horizon=control_horizon,
    noise_handling="none",
    l2_regularization=20.0,
    noise_strength=0.01,
    output_cost=1000.0,
    affine=True,
    clip_control=(-15, 15),
)

pendulum = DoubleInvertedPendulum(dt=0.01)

# initial state
pendulum.q[0] = np.pi - 0.05
pendulum.q[1] = np.pi + 0.2

n_steps = 1000
for k in range(n_steps):
    # decide on control magnitude
    controller.feed(pendulum.q)
    control_plan = controller.plan()

    # run the model
    pendulum.step(control_plan[0])

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
        ax.set_ylabel(["$\\theta_1$", "$\\theta_2$"][i])

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

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

from ddc import GeneralSystem, DeepControl

# %% [markdown]
# ## Define the simulators


class DoubleCartPole(GeneralSystem):
    def __init__(
        self,
        M: float = 2.0,
        m1: float = 1.0,
        m2: float = 0.3,
        l1: float = 0.5,
        l2: float = 0.4,
        g: float = 9.8,
        dt: float = 0.01,
        **kwargs,
    ):
        # parameters
        self.M = M
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.dt = dt

        # state: x, theta_1, theta_2, dx, dtheta_1, dtheta_2
        super().__init__(evolution=self._step, state_dim=6, control_dim=1, **kwargs)

    def _step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Apply the given force to the pendulum and update the state by one time step."""
        assert state.squeeze().shape == (6,)
        assert control.numel() == 1

        Mtot = self.M + self.m1 + self.m2
        m = self.m1 + self.m2

        theta1 = state[1]
        theta2 = state[2]
        c1 = torch.cos(theta1).item()
        s1 = torch.sin(theta1).item()
        c2 = torch.cos(theta2).item()
        s2 = torch.sin(theta2).item()

        dtheta = theta2 - theta1
        cdiff = torch.cos(dtheta).item()
        sdiff = torch.sin(dtheta).item()

        dq1sq = state[4].item() ** 2
        dq2sq = state[5].item() ** 2

        f = control.item()

        dtype = state.dtype
        A = torch.tensor(
            [
                [Mtot, self.l1 * m * c1, self.m2 * self.l2 * c2],
                [m * c1, m * self.l1, self.m2 * self.l2 * cdiff],
                [c2, self.l1 * cdiff, self.l2],
            ],
            dtype=dtype,
        )
        b = torch.tensor(
            [
                [f + m * self.l1 * dq1sq * s1 + self.m2 * self.l2 * dq2sq * s2],
                [m * self.g * s1 + self.m2 * self.l2 * dq2sq * sdiff],
                [self.g * s2 - self.l1 * dq1sq * sdiff],
            ],
            dtype=dtype,
        )

        result = torch.linalg.lstsq(A, b)
        ddq = result.solution.squeeze()

        # leap frog
        new_state = state.clone()
        change = (0.5 * self.dt) * ddq[:, None]

        new_state[3:] += change
        new_state[:3] += self.dt * new_state[3:]
        new_state[3:] += change

        return new_state


class DoubleInvertedPendulum(GeneralSystem):
    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 0.3,
        l1: float = 0.5,
        l2: float = 0.4,
        g: float = 9.8,
        dt: float = 0.01,
        **kwargs,
    ):
        # parameters
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.dt = dt

        # state: theta_1, theta_2, dtheta_1, dtheta_2
        super().__init__(evolution=self._step, state_dim=4, control_dim=2, **kwargs)

    def _step(self, state: torch.Tensor, control: torch.Tensor):
        """Apply the given torques to the pendulums and update the state by one time
        step.
        """
        assert state.squeeze().shape == (4,)
        assert control.squeeze().shape == (2,)

        m = self.m1 + self.m2

        theta1 = state[0]
        theta2 = state[1]
        c1 = torch.cos(theta1).item()
        s1 = torch.sin(theta1).item()
        c2 = torch.cos(theta2).item()
        s2 = torch.sin(theta2).item()

        dtheta = theta2 - theta1
        cdiff = torch.cos(dtheta).item()
        sdiff = torch.sin(dtheta).item()

        dq1sq = state[2].item() ** 2
        dq2sq = state[3].item() ** 2

        dtype = state.dtype
        A = torch.tensor(
            [
                [m * self.l1, self.m2 * self.l2 * cdiff],
                [self.l1 * cdiff, self.l2],
            ],
            dtype=dtype,
        )
        b = (
            torch.tensor(
                [
                    [m * self.g * s1 + self.m2 * self.l2 * dq2sq * sdiff],
                    [self.g * s2 - self.l1 * dq1sq * sdiff],
                ],
                dtype=dtype,
            )
            + control
        )

        result = torch.linalg.lstsq(A, b)
        ddq = result.solution.squeeze()

        # leap frog
        new_state = state.clone()
        change = (0.5 * self.dt) * ddq[:, None]
        new_state[2:] += change
        new_state[:2] += self.dt * new_state[2:]
        new_state[2:] += change

        return new_state


# %% [markdown]
# ### Test the simulation of double cart pole

# %%
pendulum = DoubleCartPole(
    observation=lambda _: _,
    initial_state=torch.tensor(
        [0.0, np.pi - 0.05, np.pi + 0.2, 0.0, 0.0, 0.0], dtype=torch.float64
    ),
)

n_steps = 1000
phase = torch.zeros((6, n_steps))

for i in range(n_steps):
    obs = pendulum.run(1)
    phase[:, i] = obs.squeeze()

# %%
with dv.FigureManager(3, 1, figsize=(6, 4), constrained_layout=True) as (_, axs):
    for i, ax in enumerate(axs):
        ax.plot(phase[i, :])
        ax.set_ylabel(["$x$", "$\\theta_1$", "$\\theta_2$"][i])

# %% [markdown]
# ### Test the simulation of double inverted pendulum

# %%
pendulum = DoubleInvertedPendulum(
    observation=lambda _: _,
    initial_state=torch.tensor(
        [np.pi - 0.05, np.pi + 0.2, 0.0, 0.0], dtype=torch.float64
    ),
)

n_steps = 1000
phase = torch.zeros((4, n_steps))

for i in range(n_steps):
    obs = pendulum.run(1)
    phase[:, i] = obs.squeeze()

# %%
with dv.FigureManager(2, 1, figsize=(6, 4), constrained_layout=True) as (_, axs):
    for i, ax in enumerate(axs):
        ax.plot(phase[i, :])
        ax.set_ylabel(["$\\theta_1$", "$\\theta_2$"][i])

# %% [markdown]
# ## Long horizon, online, double cart pole

# %%
torch.manual_seed(41)
control_horizon = 16
controller = DeepControl(
    control_dim=1,
    ini_length=2,
    horizon=control_horizon,
    l2_regularization=1e-5,
    control_cost=1e-4,
    seed_noise_norm=0.1,
    affine=True,
    control_norm_clip=2.0,
)

n_steps = 1000
dt = 0.01

# initial state
model = DoubleCartPole(
    dt=dt,
    observation=lambda state: state[1:3],
    initial_state=torch.tensor(
        [0.0, np.pi - 0.05, np.pi + 0.2, 0.0, 0.0, 0.0], dtype=torch.float64
    ),
)

planning_steps = 5
seed_length = 120

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
torch.manual_seed(41)
control_horizon = 16
controller = DeepControl(
    control_dim=2,
    ini_length=2,
    horizon=control_horizon,
    l2_regularization=1e-5,
    control_cost=1e-4,
    seed_noise_norm=0.1,
    affine=True,
    control_norm_clip=2.0,
)

n_steps = 1000
dt = 0.01

# initial state
model = DoubleInvertedPendulum(
    dt=dt,
    observation=lambda state: state[:2],
    initial_state=torch.tensor(
        [np.pi - 0.05, np.pi + 0.2, 0.0, 0.0], dtype=torch.float64
    ),
)

planning_steps = 5
seed_length = 120

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
    axs[-1].set_ylabel("controls")

# %%

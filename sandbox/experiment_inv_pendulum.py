# %% [markdown]
# # Try to control an inverted pendulum

# %%
# enable autoreload if we're running interactively

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
# ## Define the system


class InvertedPendulum(GeneralSystem):
    def __init__(self, dt: float, w0: float = 1.0, observation=None, **kwargs):
        self.dt = dt
        self.w0 = w0

        evolution = lambda state: torch.tensor(
            [
                state[0] + self.dt * state[1],
                state[1] + self.dt * self.w0**2 * torch.sin(state[0]),
            ],
            dtype=self.dtype,
        )
        control = lambda u: torch.tensor([0.0, self.dt * u], dtype=self.dtype)
        super().__init__(evolution, control, observation, state_dim=2, **kwargs)


# %% [markdown]
# ## Try the controller

# %%
torch.manual_seed(41)
control_horizon = 16
controller = DeepControl(
    control_dim=1,
    ini_length=2,
    horizon=control_horizon,
    l2_regularization=1e-4,
    control_cost=1e-5,
    seed_noise_norm=1.0,
    affine=True,
)

n_steps = 200
dt = 0.01

# initial state
model = InvertedPendulum(
    dt=dt,
    observation=lambda state: state[[0]],
    initial_state=torch.tensor([3.1, 0.0], dtype=torch.float64),
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
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    yl = (min(observations.min(), 0), max(observations.max(), 0))
    axs[0].axhline(0, c="gray", ls=":", lw=1.0)
    axs[0].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="seed",
    )
    axs[0].plot(observations.squeeze(), lw=1.0)
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("angle")
    # axs[0].legend(frameon=False)

    yl = (min(controls.min(), 0), max(controls.max(), 0))
    axs[1].axhline(0, c="gray", ls=":", lw=1.0)
    axs[1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="seed",
    )
    axs[1].plot(controls.squeeze(), lw=1.0)
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    # axs[1].legend(frameon=False, loc="lower right")

# %%

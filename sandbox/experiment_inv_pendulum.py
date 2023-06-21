# %% [markdown]
# # Try to control an inverted pendulum

# %%
# enable autoreload if we're running interactively

import sys

if hasattr(sys, "ps1"):
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")

        print("autoreload active")
    except ModuleNotFoundError:
        pass

# %%
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

from ddc import DDController

# %% [markdown]
# ## Short horizon, online

# %%
torch.manual_seed(42)
history_length = 12
control_horizon = 4
controller = DDController(
    1,
    1,
    history_length,
    seed_length=2,
    control_horizon=control_horizon,
    noise_handling="average",
    output_cost=0.1,
)

n_steps = 1200
dt = 0.05

# initial state
phi = 3.0
omega = 0.0

for k in range(n_steps):
    # decide on control magnitude
    controller.feed(torch.tensor([phi]))
    control_plan = controller.plan()

    # run the model
    omega += dt * (np.sin(phi) + control_plan[0].item())
    phi += dt * omega

control_start = controller.minimal_history

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    axs[0].set_title("Short horizon, online")

    yl = (outputs.min(), outputs.max())
    axs[0].axhline(0, c="gray", ls=":", lw=1.0)
    axs[0].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[0].plot(outputs.squeeze(), lw=1.0)
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("angle")
    # axs[0].legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[1].axhline(0, c="gray", ls=":", lw=1.0)
    axs[1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[1].plot(controls.squeeze(), lw=1.0)
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    # axs[1].legend(frameon=False, loc="lower right")

# %% [markdown]
# ## Short horizon, offline

# %%
torch.manual_seed(42)
history_length = 12
control_horizon = 4
controller = DDController(
    1,
    1,
    history_length,
    seed_length=2,
    control_horizon=control_horizon,
    noise_handling="average",
    output_cost=0.1,
    offline=True,
)

n_steps = 1200
dt = 0.05

# initial state
phi = 3.0
omega = 0.0

for k in range(n_steps):
    # decide on control magnitude
    controller.feed(torch.tensor([phi]))
    control_plan = controller.plan()

    # run the model
    omega += dt * (np.sin(phi) + control_plan[0].item())
    phi += dt * omega

control_start = controller.minimal_history

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    axs[0].set_title("Short horizon, offline")

    yl = (outputs.min(), outputs.max())
    axs[0].axhline(0, c="gray", ls=":", lw=1.0)
    axs[0].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[0].plot(outputs.squeeze(), lw=1.0)
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("angle")
    # axs[0].legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[1].axhline(0, c="gray", ls=":", lw=1.0)
    axs[1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[1].plot(controls.squeeze(), lw=1.0)
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    # axs[1].legend(frameon=False, loc="lower right")

# %% [markdown]
# ## Long horizon, online

# %%
torch.manual_seed(42)
history_length = 12
control_horizon = 8
controller = DDController(
    1,
    1,
    history_length,
    seed_length=2,
    control_horizon=control_horizon,
    noise_handling="svd",
    output_cost=0.1,
)

n_steps = 200
dt = 0.05

# initial state
phi = 3.0
omega = 0.0

for k in range(n_steps):
    # decide on control magnitude
    controller.feed(torch.tensor([phi]))
    control_plan = controller.plan()

    # run the model
    omega += dt * (np.sin(phi) + control_plan[0].item())
    phi += dt * omega

control_start = controller.minimal_history

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    axs[0].set_title("Long horizon, online")

    yl = (outputs.min(), outputs.max())
    axs[0].axhline(0, c="gray", ls=":", lw=1.0)
    axs[0].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[0].plot(outputs.squeeze(), lw=1.0)
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("angle")
    # axs[0].legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[1].axhline(0, c="gray", ls=":", lw=1.0)
    axs[1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[1].plot(controls.squeeze(), lw=1.0)
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    # axs[1].legend(frameon=False, loc="lower right")

# %% [markdown]
# ## Long horizon, offline

# %%
torch.manual_seed(42)
history_length = 12
control_horizon = 8
controller = DDController(
    1,
    1,
    history_length,
    seed_length=2,
    control_horizon=control_horizon,
    noise_handling="average",
    output_cost=0.1,
    offline=True,
)

n_steps = 200
dt = 0.05

# initial state
phi = 3.0
omega = 0.0

for k in range(n_steps):
    # decide on control magnitude
    controller.feed(torch.tensor([phi]))
    control_plan = controller.plan()

    # run the model
    omega += dt * (np.sin(phi) + control_plan[0].item())
    phi += dt * omega

control_start = controller.minimal_history

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    axs[0].set_title("Long horizon, offline")

    yl = (outputs.min(), outputs.max())
    axs[0].axhline(0, c="gray", ls=":", lw=1.0)
    axs[0].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[0].plot(outputs.squeeze(), lw=1.0)
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("angle")
    # axs[0].legend(frameon=False)

    yl = (controls.min(), controls.max())
    axs[1].axhline(0, c="gray", ls=":", lw=1.0)
    axs[1].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="no control",
    )
    axs[1].plot(controls.squeeze(), lw=1.0)
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    # axs[1].legend(frameon=False, loc="lower right")

# %%

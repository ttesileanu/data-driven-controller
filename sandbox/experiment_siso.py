# %% [markdown]
## Experiments with controlling SISO linear systems

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

from ddc import LinearSystem, DDController

# %% [markdown]
# ## Noiseless

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
)
history_length = 25
control_horizon = 4
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
outputs = [model.observe()[:, 0]]
controls = []
for k in range(n_steps):
    # need noise for exploration
    control = control + control_noise * torch.randn(control.shape)
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(control, y)
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
# ## With observation noise

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    observation_noise=torch.tensor([[0.2]]),
)
history_length = 25
control_horizon = 4
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
outputs = [model.observe()[:, 0]]
controls = []
for k in range(n_steps):
    # need noise for exploration
    control = control + control_noise * torch.randn(control.shape)
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(control, y)
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

# %%

# %% [markdown]
## Experiments with controlling SISO linear systems, offline method

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
).convert_type(torch.float64)
history_length = 25
control_horizon = 1
controller = DDController(
    1,
    1,
    history_length,
    control_horizon=control_horizon,
    offline=True,
)

n_steps = 200
y = model.observe()[:, 0]
for k in range(n_steps):
    controller.feed(y)
    control_plan = controller.plan()
    y = model.run(control_plan=control_plan[[0]])
    y = y[0, :, 0]

control_start = controller.minimal_history

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

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
# ## With observation noise, naive controller

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    observation_noise=torch.tensor([[0.01]]),
).convert_type(torch.float64)
history_length = 25
control_horizon = 1
controller = DDController(
    1,
    1,
    history_length,
    control_horizon=control_horizon,
    offline=True,
)

n_steps = 200
y = model.observe()[:, 0]
for k in range(n_steps):
    controller.feed(y)
    control_plan = controller.plan()
    y = model.run(control_plan=control_plan[[0]])
    y = y[0, :, 0]

control_start = controller.minimal_history

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

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
# ## With observation noise, averaging controller

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    observation_noise=torch.tensor([[0.1]]),
).convert_type(torch.float64)
history_length = 30
control_horizon = 1
controller = DDController(
    1,
    1,
    history_length,
    control_horizon=control_horizon,
    noise_handling="average",
    offline=True,
)

n_steps = 300
y = model.observe()[:, 0]
for k in range(n_steps):
    controller.feed(y)
    control_plan = controller.plan()
    y = model.run(control_plan=control_plan[[0]])
    y = y[0, :, 0]

control_start = controller.minimal_history

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

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
# ## With sparsity

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    observation_noise=torch.tensor([[0.1]]),
).convert_type(torch.float64)
history_length = 30
control_horizon = 1
controller = DDController(
    1,
    1,
    history_length,
    control_horizon=control_horizon,
    control_sparsity=0.1,
    method="gd",
    gd_lr=1e-3,
    gd_iterations=150,
    noise_handling="average",
    offline=True,
)

n_steps = 300
y = model.observe()[:, 0]
for k in range(n_steps):
    controller.feed(y)
    control_plan = controller.plan()
    y = model.run(control_plan=control_plan[[0]])
    y = y[0, :, 0]

control_start = controller.minimal_history

outputs = torch.stack(controller.history.outputs)
controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = torch.stack(controller.history.controls)

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

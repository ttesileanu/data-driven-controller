# %% [markdown]
# # Experiments with sparse controllers

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
# ## A 1d example with noise; no sparsity, no control cost, lstsq method

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    state_noise=torch.tensor([[0.05]]),
    observation_noise=torch.tensor([[0.05]]),
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

n_steps = 400
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
# ## A 1d example with noise; no sparsity, no control cost, gd method

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    state_noise=torch.tensor([[0.05]]),
    observation_noise=torch.tensor([[0.05]]),
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
    method="gd",
    gd_lr=1e-5,
    gd_iterations=75,
)

n_steps = 400
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
# ## A 1d example with noise and control cost; no sparsity, lstsq method

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    state_noise=torch.tensor([[0.05]]),
    observation_noise=torch.tensor([[0.05]]),
)
history_length = 25
control_horizon = 5
controller = DDController(
    1,
    1,
    history_length,
    control_horizon=control_horizon,
    control_cost=0.01,
    target_cost=0.1,
)

n_steps = 400
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
with dv.FigureManager() as (_, ax):
    ax.axhline(0, c="gray", ls=":", lw=1)
    ax.axvline(0, c="gray", ls=":", lw=1)
    ax.scatter(outputs[:-1], raw_controls[1:], s=10, alpha=0.7)
    ax.set_xlabel("measurement $y_t$")
    ax.set_ylabel("control $u_t$")

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    yl = (outputs.min(), outputs.max())
    axs[0].axhline(0, c="gray", lw=1.0, ls=":")
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
    axs[0].set_yscale("symlog", linthresh=1)
    # axs[0].set_ylim(-10, 10)

    yl = (controls.min(), controls.max())
    axs[1].axhline(0, c="gray", lw=1.0, ls=":")
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
    axs[1].set_yscale("symlog", linthresh=1)
    # axs[1].set_ylim(-10, 10)

# %% [markdown]
# ## A 1d example with control cost and only observer noise; no sparsity, lstsq method

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    observation_noise=torch.tensor([[0.05]]),
)
history_length = 25
control_horizon = 5
controller = DDController(
    1,
    1,
    history_length,
    control_horizon=control_horizon,
    control_cost=0.01,
    target_cost=0.1,
)

n_steps = 1000
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
with dv.FigureManager() as (_, ax):
    ax.axhline(0, c="gray", ls=":", lw=1)
    ax.axvline(0, c="gray", ls=":", lw=1)
    ax.scatter(outputs[:-1], controls_prenoise[1:], s=10, alpha=0.7)
    ax.set_xlabel("measurement $y_t$")
    ax.set_ylabel("control $u_t$")

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    yl = (outputs.min(), outputs.max())
    axs[0].axhline(0, c="gray", lw=1.0, ls=":")
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
    axs[0].set_yscale("symlog", linthresh=1)
    # axs[0].set_ylim(-10, 10)

    yl = (controls.min(), controls.max())
    axs[1].axhline(0, c="gray", lw=1.0, ls=":")
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
    axs[1].set_yscale("symlog", linthresh=1)
    # axs[1].set_ylim(-10, 10)

# %% [markdown]
# ## Same as above but with short horizon

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    observation_noise=torch.tensor([[0.05]]),
)
history_length = 25
control_horizon = 2
controller = DDController(
    1,
    1,
    history_length,
    control_horizon=control_horizon,
    control_cost=0.01,
    target_cost=0.1,
)

n_steps = 1000
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
with dv.FigureManager() as (_, ax):
    ax.axhline(0, c="gray", ls=":", lw=1)
    ax.axvline(0, c="gray", ls=":", lw=1)
    ax.scatter(outputs[:-1], controls_prenoise[1:], s=10, alpha=0.7)
    ax.set_xlabel("measurement $y_t$")
    ax.set_ylabel("control $u_t$")

# %%
with dv.FigureManager(2, 1, figsize=(6, 4)) as (_, axs):
    yl = (outputs.min(), outputs.max())
    axs[0].axhline(0, c="gray", lw=1.0, ls=":")
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
    axs[0].set_yscale("symlog", linthresh=1)
    # axs[0].set_ylim(-10, 10)

    yl = (controls.min(), controls.max())
    axs[1].axhline(0, c="gray", lw=1.0, ls=":")
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
    axs[1].set_yscale("symlog", linthresh=1)
    # axs[1].set_ylim(-10, 10)

# %% [markdown]
# ## A 1d example with noise and control cost; no sparsity, gd method

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    state_noise=torch.tensor([[0.05]]),
    observation_noise=torch.tensor([[0.05]]),
)
history_length = 25
control_horizon = 5
controller = DDController(
    1,
    1,
    history_length,
    control_horizon=control_horizon,
    control_cost=0.01,
    target_cost=0.1,
    method="gd",
    gd_lr=1e-6,
    gd_iterations=75,
)

n_steps = 400
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
# ## A 1d example with sparsity

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
    control_cost=0.01,
    target_cost=0.1,
    control_sparsity=5.0,
    method="gd",
    gd_lr=1e-4,
    gd_iterations=75,
)

n_steps = 100
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

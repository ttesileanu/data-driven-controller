# %% [markdown]
## Experiments with controlling linear systems

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
### Try the simulator
#### One-dimensional, no noise

# %%
model = LinearSystem(
    evolution=torch.Tensor([[0.9]]),
    control=torch.Tensor([[1.0]]),
    initial_state=torch.Tensor([1.0]),
)
observations = model.run(100, store_initial=True)

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
observations = model.run(100, store_initial=True)

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
observations = model.run(100, store_initial=True)

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
observations = model.run(control_plan=control_plan, store_initial=True)

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
observations = model.run(control_plan=control_plan, store_initial=True)

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
observations = model.run(control_plan=control_plan, store_initial=True)

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
observations = model.run(control_plan=control_plan, store_initial=True)

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

# %%
rot_mat = torch.FloatTensor([[np.cos(0.2), np.sin(0.2)], [-np.sin(0.2), np.cos(0.2)]])
model = LinearSystem(
    evolution=0.99 * rot_mat,
    control=torch.eye(2),
    initial_state=torch.tensor([1.0, 0.0]),
)
observations = model.run(25, store_initial=True)

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

# %%
rot_mat = torch.FloatTensor([[np.cos(0.2), np.sin(0.2)], [-np.sin(0.2), np.cos(0.2)]])
model = LinearSystem(
    evolution=0.99 * rot_mat,
    control=torch.eye(2),
    observation_noise=0.001 * torch.tensor([[0.55, -0.45], [-0.45, 0.55]]),
    initial_state=torch.tile(torch.tensor([[1.0], [0.0]]), (1, 25)),
)
observations = model.run(25, store_initial=True)

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

# %%
rot_mat = torch.FloatTensor([[np.cos(0.2), np.sin(0.2)], [-np.sin(0.2), np.cos(0.2)]])
model = LinearSystem(
    evolution=0.99 * rot_mat,
    control=torch.eye(2),
    state_noise=0.0001 * torch.tensor([[0.55, -0.45], [-0.45, 0.55]]),
    initial_state=torch.tile(torch.tensor([[1.0], [0.0]]), (1, 25)),
)
observations = model.run(25, store_initial=True)

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

# %%
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
observations = model.run(control_plan=control_plan, store_initial=True)

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
# ## Try the data-driven controller
# ### A 1d example

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
    noise_handling="average",
)

n_steps = 100
control = torch.tensor([0.0])
control_noise = 0.001
outputs = [model.observe()[:, 0]]
controls = []
for k in range(n_steps):
    # need noise for exploration
    eps = control_noise * (2 * torch.rand(control.shape) - 1)
    control = control + eps * torch.linalg.norm(outputs[-1])
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(control, y)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = controller.minimal_history

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
# ### A 2d example

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
    noise_handling="average",
)

n_steps = 200
control = torch.tensor([0.0])
control_noise = 0.001
outputs = [model.observe()[:, 0]]
controls = []
for k in range(n_steps):
    # need noise for exploration
    eps = control_noise * (2 * torch.rand(control.shape) - 1)
    control = control + eps * torch.linalg.norm(outputs[-1])
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(control, y)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = controller.minimal_history

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
# ### 2d with rotation and observation noise

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
    noise_handling="average",
)

n_steps = 200
control = torch.tensor([0.0])
control_noise = 0.001
outputs = [model.observe()[:, 0]]
controls = []
for k in range(n_steps):
    # need noise for exploration
    eps = control_noise * (2 * torch.rand(control.shape) - 1)
    control = control + eps * torch.linalg.norm(outputs[-1])
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(control, y)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = controller.minimal_history

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
# ### 2d with rotation and state noise

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
    noise_handling="average",
)

n_steps = 200
control = torch.tensor([0.0])
control_noise = 0.001
outputs = [model.observe()[:, 0]]
controls = []
for k in range(n_steps):
    # need noise for exploration
    eps = control_noise * (2 * torch.rand(control.shape) - 1)
    control = control + eps * torch.linalg.norm(outputs[-1])
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(control, y)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = controller.minimal_history

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
# ### 2d rotation, observation noise, state noise, and partial observation

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
    noise_handling="average",
)

n_steps = 200
control = torch.tensor([0.0])
control_noise = 0.001
outputs = [model.observe()[:, 0]]
controls = []
for k in range(n_steps):
    # need noise for exploration
    eps = control_noise * (2 * torch.rand(control.shape) - 1)
    control = control + eps * torch.linalg.norm(outputs[-1])
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(control, y)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = controller.minimal_history

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

# %% [markdown]
# ### 2d rotation, observation noise, state noise, and partial observation

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
    noise_handling="average",
)

n_steps = 2500
control = torch.tensor([0.0])
control_noise = 0.001
outputs = [model.observe()[:, 0]]
controls = []
for k in range(n_steps):
    # need noise for exploration
    eps = control_noise * (2 * torch.rand(control.shape) - 1)
    control = control + eps * torch.linalg.norm(outputs[-1])
    y = model.run(control_plan=control[None, :])
    y = y[0, :, 0]

    outputs.append(y)
    controls.append(control)

    controller.feed(control, y)
    control_plan = controller.plan()
    control = control_plan[0]

control_start = controller.minimal_history

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

# %% [markdown]
## Experiments with controlling SISO linear systems

# %%
# enable autoreload
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

from ddc import LinearSystem, DeepControl

# %% [markdown]
# ## Noiseless

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
).convert_type(torch.float64)
controller = DeepControl(control_dim=1, horizon=10)

n_steps = 200
planning_steps = 5
seed_length = 20

control_snippet = controller.generate_seed(seed_length, torch.float64)
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
    yl = (observations.min(), observations.max())
    axs[0].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="seed",
    )
    axs[0].plot(observations.squeeze())
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
        label="seed",
    )
    axs[1].plot(controls.squeeze())
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    axs[1].legend(frameon=False, loc="lower right")

# %% [markdown]
# ## With observation noise, offline

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    observation_noise=torch.tensor([[0.1]]),
).convert_type(torch.float64)
controller = DeepControl(control_dim=1, horizon=15, l2_regularization=2.0, online=False)

n_steps = 1000
planning_steps = 4
seed_length = 40

control_snippet = controller.generate_seed(seed_length, torch.float64)
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
    yl = (observations.min(), observations.max())
    axs[0].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="seed",
    )
    axs[0].plot(observations.squeeze())
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
        label="seed",
    )
    axs[1].plot(controls.squeeze())
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    axs[1].legend(frameon=False, loc="lower right")

# %% [markdown]
# ## With observation noise, online

# %%
torch.manual_seed(42)
model = LinearSystem(
    evolution=torch.tensor([[1.1]]),
    control=torch.tensor([[1.0]]),
    initial_state=torch.tensor([[1.0]]),
    observation_noise=torch.tensor([[0.1]]),
).convert_type(torch.float64)
controller = DeepControl(
    control_dim=1, horizon=15, l2_regularization=2.0, online=True, control_cost=0.1
)

n_steps = 1000
planning_steps = 4
seed_length = 40

control_snippet = controller.generate_seed(seed_length, torch.float64)
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
    yl = (observations.min(), observations.max())
    axs[0].fill_betweenx(
        yl,
        [0, 0],
        2 * [control_start],
        color="gray",
        alpha=0.5,
        edgecolor="none",
        label="seed",
    )
    axs[0].plot(observations.squeeze())
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
        label="seed",
    )
    axs[1].plot(controls.squeeze())
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("control")
    axs[1].legend(frameon=False, loc="lower right")

# %%

# %% [markdown]
# # Trying to see if our methods can work in the affine case

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

from ddc import DeepControl
from ddc.solvers import lstsq_constrained

# %% [markdown]
# ## Check that simply increasing size of lag vector should work in principle

# %%
a = 0.8
phi = 0.9

x_init = 0.0
T = 30
x = torch.zeros(T, dtype=torch.float64)
x[0] = x_init
for t in range(T - 1):
    x[t + 1] = a * x[t] + phi

# %%
with dv.FigureManager() as (_, ax):
    ax.plot(x)

# %%
Xm = x[None, :-2]
X = x[None, 1:-1]
Xp = x[None, 2:]

Z = torch.vstack((Xm, X))
Zpseudo = torch.linalg.pinv(Z)

Zpseudo_again = Z.T @ torch.linalg.inv(Z @ Z.T)
assert (Zpseudo - Zpseudo_again).abs().max() < 1e-9

xp = 1.7
xpp = (xp - phi) / a

g = Zpseudo @ torch.tensor([[xpp], [xp]], dtype=torch.float64)

xf = Xp @ g

expected_xf = a * xp + phi
assert (xf - expected_xf).abs().max() < 1e-9

# %% [markdown]
# ### Try this with our controller

# %%
torch.manual_seed(42)
control_horizon = 4
controller = DeepControl(
    control_dim=1,
    ini_length=2,
    horizon=control_horizon,
    l2_regularization=1.0,
    control_cost=1e-3,
    noise_strength=0.01,
    relative_noise=False,
)

n_steps = 200

# initial state
state = 0.0

seed_length = 12

control_snippet = controller.generate_seed(seed_length)
observation_snippet = torch.zeros((seed_length, 1))
for k in range(seed_length):
    observation_snippet[k, 0] = state
    state = a * state + phi + control_snippet[k]

control_start = len(control_snippet)
controller.feed(observation_snippet)

for k in range(n_steps):
    # decide on control magnitude
    control_plan = controller.plan()

    u = control_plan[0]
    controller.feed(torch.tensor([[state]]))

    # run the model
    state = a * state + phi + u.item()

outputs = torch.stack(controller.history.observations)
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
# ## Check that adding sum(alpha) = 1 constraint should work in principle

# %%
a = 0.8
phi = 0.9

x_init = 0.0
T = 30
x = torch.zeros(T, dtype=torch.float64)
x[0] = x_init
for t in range(T - 1):
    x[t + 1] = a * x[t] + phi

# %%
with dv.FigureManager() as (_, ax):
    ax.plot(x)

# %%
X = x[None, :-1]
Xf = x[None, 1:]

xp = 1.7

M = torch.ones((1, X.shape[1]), dtype=x.dtype)

g = lstsq_constrained(
    X, torch.tensor([[xp]], dtype=x.dtype), M, torch.tensor([[1.0]], dtype=x.dtype)
)

xf = Xf @ g

expected_xf = a * xp + phi
assert (xf - expected_xf).abs().max() < 1e-9

# %% [markdown]
# ### Try this with our controller

# %%
torch.manual_seed(42)
control_horizon = 4
controller = DeepControl(
    control_dim=1,
    ini_length=1,
    horizon=control_horizon,
    l2_regularization=1.0,
    control_cost=1e-3,
    noise_strength=0.01,
    relative_noise=False,
    affine=True,
)

seed_length = 12

control_snippet = controller.generate_seed(seed_length)
observation_snippet = torch.zeros((seed_length, 1))
for k in range(seed_length):
    observation_snippet[k, 0] = state
    state = a * state + phi + control_snippet[k]

control_start = len(control_snippet)
controller.feed(observation_snippet)

for k in range(n_steps):
    # decide on control magnitude
    control_plan = controller.plan()

    u = control_plan[0]
    controller.feed(torch.tensor([[state]]))

    # run the model
    state = a * state + phi + u.item()

outputs = torch.stack(controller.history.observations)
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

# %%

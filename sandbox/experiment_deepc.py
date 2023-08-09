# %% [markdown]
# # Compare DeePC with our implementation

# %%
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydeepc import DeePC
from pydeepc.utils import Data
from deepc_utils import System

import torch
from ddc import DeepControl

# %% [markdown]
# ## DeePC on SISO pulley problem
# XXX should figure out what this problem is


# %%
# Define the loss function for DeePC
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Sum_t ||y_t - 1||^2
    return 1e3 * cp.norm(y - 1, "fro") ** 2 + 1e-1 * cp.norm(u, "fro") ** 2


# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of input/output constraints
    return []


# %%
# DeePC parameters
s = 1  # How many steps before we solve again the DeePC problem
T_INI = 4  # Size of the initial set of data
T_list = [50, 500]  # Number of data points used to estimate the system
HORIZON = 10  # Horizon length
LAMBDA_G_REGULARIZER = 0.1  # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 0  # y regularizer (see DeePC paper, eq. 8)
LAMBDA_U_REGULARIZER = 0  # u regularizer
EXPERIMENT_HORIZON = 100  # Total number of steps

# Plant
# In this example we consider the three-pulley
# system analyzed in the original VRFT paper:
#
# "Virtual reference feedback tuning:
#      a direct method for the design offeedback controllers"
# -- Campi et al. 2003

# %%
dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = System(scipysig.TransferFunction(num, den, dt=dt).to_ss())

fig, ax = plt.subplots(1, 2)
plt.margins(x=0, y=0)


# Simulate for different values of T
for T in T_list:
    print(f"Simulating with {T} initial samples...")
    sys.reset()
    # Generate initial data and initialize DeePC
    data = sys.apply_input(u=np.random.normal(size=T).reshape((T, 1)), noise_std=0)
    deepc = DeePC(data, Tini=T_INI, horizon=HORIZON)

    # Create initial data
    data_ini = Data(u=np.zeros((T_INI, 1)), y=np.zeros((T_INI, 1)))
    sys.reset(data_ini=data_ini)

    deepc.build_problem(
        build_loss=loss_callback,
        build_constraints=constraints_callback,
        lambda_g=LAMBDA_G_REGULARIZER,
        lambda_y=LAMBDA_Y_REGULARIZER,
        lambda_u=LAMBDA_U_REGULARIZER,
    )

    for idx in range(EXPERIMENT_HORIZON // s):
        # Solve DeePC
        u_optimal, info = deepc.solve(data_ini=data_ini, warm_start=True)

        # Apply optimal control input
        _ = sys.apply_input(u=u_optimal[:s, :], noise_std=1e-2)

        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)

    # Plot curve
    data = sys.get_all_samples()
    ax[0].plot(data.y[T_INI:], label=f"$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$")
    ax[1].plot(data.u[T_INI:], label=f"$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$")

ax[0].set_ylim(0, 2)
ax[1].set_ylim(-4, 4)
ax[0].set_xlabel("t")
ax[0].set_ylabel("y")
ax[0].grid()
ax[1].set_ylabel("u")
ax[1].set_xlabel("t")
ax[1].grid()
ax[0].set_title("Closed loop - output signal $y_t$")
ax[1].set_title("Closed loop - control signal $u_t$")
ax[0].legend(fancybox=True, shadow=True)
plt.show()

# %% [markdown]
# ## Run our implementation on SISO pulley problem

# %%
# %%
dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = System(scipysig.TransferFunction(num, den, dt=dt).to_ss())

fig, ax = plt.subplots(1, 2, constrained_layout=True)
for a in ax:
    a.margins(x=0, y=0)


# Simulate for different values of T
for T in T_list:
    print(f"Simulating with {T} initial samples...")
    sys.reset()

    # Initialize controller
    controller = DeepControl(
        1,
        ini_length=T_INI,
        horizon=HORIZON,
        online=False,
        l2_regularization=1e-4,
        control_cost=1e-4,
        target=1.0,
    )

    control_snippet = controller.generate_seed(T, torch.float64)
    _ = sys.apply_input(u=control_snippet.numpy(), noise_std=0.0)
    controller.feed(torch.from_numpy(sys.y))

    # Create initial data
    data_ini = Data(u=np.zeros((T_INI, 1)), y=np.zeros((T_INI, 1)))
    sys.reset(data_ini=data_ini)

    for idx in range(EXPERIMENT_HORIZON + T):
        control_plan = controller.plan()
        control_snippet = control_plan[:s]

        # Apply optimal control input
        _ = sys.apply_input(u=control_snippet.numpy(), noise_std=0.0)

        observation_snippet = torch.from_numpy(sys.get_last_n_samples(s).y)
        controller.feed(observation_snippet)

    # Plot curve
    data = sys.get_all_samples()
    ax[0].plot(data.y[T_INI:], label=f"$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$")
    ax[1].plot(data.u[T_INI:], label=f"$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$")

ax[0].set_ylim(0, 2)
ax[1].set_ylim(-4, 4)
ax[0].set_xlabel("t")
ax[0].set_ylabel("y")
ax[0].grid(ls=":")
ax[1].set_ylabel("u")
ax[1].set_xlabel("t")
ax[1].grid(ls=":")
ax[0].set_title("Closed loop - output signal $y_t$")
ax[1].set_title("Closed loop - control signal $u_t$")
ax[0].legend()

for a in ax:
    sns.despine(ax=a, offset=10)

# %% [mardown]
# ## DeePC on MIMO two tank


# Define the loss function for DeePC
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    ref = np.ones(y.shape)
    return cp.norm(y - ref, "fro") ** 2


# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of input/output constraints
    # no real constraints on y, input should be between -1 and 1
    return [u >= -1, u <= 1]


# DeePC paramters
s = 1  # How many steps before we solve again the DeePC problem
T_INI = 2  # Size of the initial set of data
T_list = [100]  # Number of data points used to estimate the system
HORIZON = 10  # Horizon length
LAMBDA_G_REGULARIZER = 0  # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 0  # y regularizer (see DeePC paper, eq. 8)
LAMBDA_U_REGULARIZER = 0  # u regularizer
EXPERIMENT_HORIZON = 100  # Total number of steps

# model of two-tank example
A = np.array([[0.70469, 0.0], [0.24664, 0.70469]])
B = np.array([[0.75937], [0.12515]])
C = np.array([[0.0, 1.0]])
D = np.zeros((C.shape[0], B.shape[1]))

sys = System(scipysig.StateSpace(A, B, C, D, dt=1))

fig, ax = plt.subplots(1, 2)
plt.margins(x=0, y=0)


# Simulate for different values of T
for T in T_list:
    print(f"Simulating with {T} initial samples...")
    sys.reset()
    # Generate initial data and initialize DeePC

    data = sys.apply_input(u=np.random.normal(size=T).reshape((T, 1)), noise_std=0)
    deepc = DeePC(data, Tini=T_INI, horizon=HORIZON)

    # Create initial data
    data_ini = Data(u=np.zeros((T_INI, 1)), y=np.zeros((T_INI, 1)))
    sys.reset(data_ini=data_ini)

    deepc.build_problem(
        build_loss=loss_callback,
        build_constraints=constraints_callback,
        lambda_g=LAMBDA_G_REGULARIZER,
        lambda_y=LAMBDA_Y_REGULARIZER,
        lambda_u=LAMBDA_U_REGULARIZER,
    )

    for _ in range(EXPERIMENT_HORIZON // s):
        # Solve DeePC
        u_optimal, info = deepc.solve(data_ini=data_ini, warm_start=True)

        # Apply optimal control input
        _ = sys.apply_input(u=u_optimal[:s, :], noise_std=1e-2)

        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)

    # Plot curve
    data = sys.get_all_samples()
    ax[0].plot(data.y[T_INI:], label=f"$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$")
    ax[1].plot(data.u[T_INI:], label=f"$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$")

ax[0].set_ylim(0, 1.5)
ax[1].set_ylim(-1.2, 1.2)
ax[0].set_xlabel("t")
ax[0].set_ylabel("y")
ax[0].grid()
ax[1].set_ylabel("u")
ax[1].set_xlabel("t")
ax[1].grid()
ax[0].set_title("Closed loop - output signal $y_t$")
ax[1].set_title("Closed loop - control signal $u_t$")
plt.legend(fancybox=True, shadow=True)
plt.show()

# %% [markdown]
# ## Run our implementation on MIMO two tank

# %%
# model of two-tank example
A = np.array([[0.70469, 0.0], [0.24664, 0.70469]])
B = np.array([[0.75937], [0.12515]])
C = np.array([[0.0, 1.0]])
D = np.zeros((C.shape[0], B.shape[1]))

sys = System(scipysig.StateSpace(A, B, C, D, dt=1))

fig, ax = plt.subplots(1, 2, constrained_layout=True)
for a in ax:
    a.margins(x=0, y=0)


# Simulate for different values of T
for T in T_list:
    print(f"Simulating with {T} initial samples...")
    sys.reset()

    # Initialize controller
    controller = DeepControl(
        1,
        ini_length=T_INI,
        horizon=HORIZON,
        online=False,
        l2_regularization=1e-4,
        control_cost=1e-4,
        target=1.0,
        control_norm_clip=1.0,
    )

    control_snippet = controller.generate_seed(T, torch.float64)
    _ = sys.apply_input(u=control_snippet.numpy(), noise_std=0.0)
    controller.feed(torch.from_numpy(sys.y))

    # Create initial data
    data_ini = Data(u=np.zeros((T_INI, 1)), y=np.zeros((T_INI, 1)))
    sys.reset(data_ini=data_ini)

    for idx in range(EXPERIMENT_HORIZON + T):
        control_plan = controller.plan()
        control_snippet = control_plan[:s]

        # Apply optimal control input
        _ = sys.apply_input(u=control_snippet.numpy(), noise_std=0.0)

        observation_snippet = torch.from_numpy(sys.get_last_n_samples(s).y)
        controller.feed(observation_snippet)

    # Plot curve
    data = sys.get_all_samples()
    ax[0].plot(data.y[T_INI:], label=f"$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$")
    ax[1].plot(data.u[T_INI:], label=f"$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$")

ax[0].set_ylim(0, 1.5)
ax[1].set_ylim(-1.2, 1.2)
ax[0].set_xlabel("t")
ax[0].set_ylabel("y")
ax[0].grid(ls=":")
ax[1].set_ylabel("u")
ax[1].set_xlabel("t")
ax[1].grid(ls=":")
ax[0].set_title("Closed loop - output signal $y_t$")
ax[1].set_title("Closed loop - control signal $u_t$")
ax[0].legend()

for a in ax:
    sns.despine(ax=a, offset=10)

# %% [markdown]
# ## Run DeePC on inverted pendulum problem


# %%
# Define the loss function for DeePC
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    return 1e3 * cp.norm(y, "fro") ** 2 + 1e-1 * cp.norm(u, "fro") ** 2


# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of input/output constraints
    return []


# %%
rng = np.random.default_rng(42)

n_steps = 150
dt = 0.05

# initial state
phi = 3.0
omega = 0.0

T_INI = 2
HORIZON = 10

LAMBDA_G_REGULARIZER = 1.0  # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 0  # y regularizer (see DeePC paper, eq. 8)
LAMBDA_U_REGULARIZER = 0  # u regularizer

n_open = 30
data = Data(np.zeros((n_steps, 1)), np.zeros((n_steps, 1)))

deepc = None
for k in range(n_steps):
    if k < n_open:
        f = rng.normal()
        omega += dt * (np.sin(phi) + f)
        phi += dt * omega

        data.u[k] = f
        data.y[k] = phi
    else:
        # if k == n_open:
        #     data_open = Data(data.u[:n_open], data.y[:n_open])
        #     deepc = DeePC(data_open, Tini=T_INI, horizon=HORIZON)

        #     deepc.build_problem(
        #         build_loss=loss_callback,
        #         build_constraints=constraints_callback,
        #         lambda_g=LAMBDA_G_REGULARIZER,
        #         lambda_y=LAMBDA_Y_REGULARIZER,
        #         lambda_u=LAMBDA_U_REGULARIZER,
        #     )
        if k == n_open or k % 10 == 0:
            data_desc = Data(data.u[-n_open:], data.y[-n_open:])
            deepc = DeePC(data_desc, Tini=T_INI, horizon=HORIZON)

            deepc.build_problem(
                build_loss=loss_callback,
                build_constraints=constraints_callback,
                lambda_g=LAMBDA_G_REGULARIZER,
                lambda_y=LAMBDA_Y_REGULARIZER,
                lambda_u=LAMBDA_U_REGULARIZER,
            )

        # Solve DeePC
        data_ini = Data(data.u[-T_INI:], data.y[-T_INI:])
        u_optimal, info = deepc.solve(data_ini=data_ini, warm_start=True)

        # Apply optimal control input
        assert s == 1
        f = u_optimal[0, :] + 1e-2 * rng.normal()
        omega += dt * (np.sin(phi) + f.item())
        phi += dt * omega

        data.u[k] = f
        data.y[k] = phi

control_start = n_open

outputs = data.y
# controls_prenoise = torch.stack(controller.history.controls_prenoise)
controls = data.u

# %%
fig, axs = plt.subplots(2, 1, figsize=(6, 4))
axs[0].set_title("DeePC on inverted pendulum")

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

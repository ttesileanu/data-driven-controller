# %% [markdown]
# # Test optimization with L1 constraint

# %%
import torch
import numpy as np
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt
import seaborn as sns
import pydove as dv

from ddc.solvers import lstsq_l1reg

# %% [markdown]
# ## Solve a lasso problem, compare to scikit-learn

# set up the problem:
# min 1/(2 * N) * ||y - Xw||_2^2 + alpha * ||w||_1
torch.manual_seed(42)
n = 13
N = 121
X = torch.randn(N, n, dtype=torch.float64)
y = torch.rand(N, 1, dtype=torch.float64)
alpha = 0.06

# solve using scikit-learn
lasso = Lasso(alpha, tol=1e-7, fit_intercept=False)
lasso.fit(X, y)

# solve using our solver
# our solver solves
# min 1/2 * ||Aw - b||_2^2 + gamma * ||Uw - c||_1
# --> A = X, b = y, gamma = alpha * N, U = eye, c = 0
U = torch.eye(n, n, dtype=torch.float64)
c = torch.zeros(n, 1, dtype=torch.float64)
my_soln = lstsq_l1reg(X, y, U, c, alpha * N, lr=0.05, max_iterations=250)

# solution with L1 regularization
nonreg_soln = torch.linalg.lstsq(X, y)

# %%
with dv.FigureManager() as (_, ax):
    ax.plot(my_soln.loss_history)

with dv.FigureManager() as (_, ax):
    ax.axline((0, 0), slope=1, ls="--", c="gray", lw=1)
    ax.scatter(lasso.coef_, my_soln.solution.squeeze())
    ax.set_xlabel("scikit-learn solution")
    ax.set_ylabel("our solution")

with dv.FigureManager() as (_, ax):
    ax.axline((0, 0), slope=1, ls="--", c="gray", lw=1)
    ax.scatter(nonreg_soln.solution, my_soln.solution.squeeze())
    ax.set_xlabel("non-regularized solution")
    ax.set_ylabel("our solution")

# %%

"""Various solvers used by the controllers."""

import torch

from types import SimpleNamespace


def lstsq_constrained(
    A: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    v: torch.Tensor,
    l2_regularization: float = 0.0,
) -> torch.Tensor:
    """Least-squares minimization with linear constraint.

    Minimize `norm(A @ x - b) ** 2` subject to `M @ x = v`, where the `norm` is the L2
    norm. This assumes that `M` is full rank.

    Based on solving system expanded with Lagrange multiplier (see Boyd and
    Vandenberghe, 2004, section 10.1.1).

    If `l2_regularization` is non-zero, an L2 regularizer is used, i.e.,

        l2_regularization * norm(x) ** 2

    is added to the objective.

    :param A: matrix involved in minimization, `min(norm(A @ x - b))`
    :param b: vector involved in minimization, `min(norm(A @ x - b))`
    :param M: matrix involved in the constraint, `M @ x = v`
    :param v: vector involved in the constraint, `M @ x = v`
    :param l2_regularization: amount of L2 regularization
    :return: the solution
    """
    # check dimensions
    n = A.shape[-1]
    m = M.shape[-2]
    assert M.shape[-1] == n
    assert b.shape[0] == A.shape[-2]
    assert v.shape[0] == m
    assert m <= n

    # if l2_regularization != 0:
    #     # easiest way to impose this: extend A and b
    #     A = torch.vstack((A, l2_regularization * torch.eye(n, dtype=A.dtype)))
    #     b = torch.vstack((b, torch.zeros((n, 1), dtype=b.dtype)))

    # regularization amounts to adding stacking an identity under A and zeros under b
    # this in turn comes down to adding a term proportional to the identity in the
    # lower-left corner of extended_A below
    ATA = A.T @ A
    if l2_regularization != 0:
        ATA += l2_regularization * torch.eye(n, dtype=ATA.dtype)
    top_row = torch.hstack((ATA, M.T))
    zero = torch.zeros(len(M), len(M), dtype=M.dtype)
    bottom_row = torch.hstack((M, zero))
    extended_A = torch.vstack((top_row, bottom_row))
    extended_b = torch.vstack((A.T @ b, v))
    zstar = torch.linalg.solve(extended_A, extended_b)

    # XXX should check that there was a solution
    # XXX should report whether the solution was unique?

    xstar = zstar[: len(top_row)]
    return xstar


def lstsq_l1reg_constrained(
    A: torch.Tensor,
    b: torch.Tensor,
    M: torch.Tensor,
    v: torch.Tensor,
    C: torch.Tensor,
    c: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """L1-regularized least-squares minimization with linear constraint.

    Minimize `norm(A @ x - b) + gamma * l1_norm(U @ x - c)` subject to `M @ x = v`,
    where the first `norm` is the L2 norm and the `l1_norm` is the L1 norm. This assumes
    that `M` is full rank.

    This performs gradient descent in the solution space of the constraint, using a
    coordinate system aligned with the singular vectors of the resulting quadratic form.

    :param A: matrix involved in minimization, `min(norm(A @ x - b))`
    :param b: vector involved in minimization, `min(norm(A @ x - b))`
    :param M: matrix involved in the constraint, `M @ x = v`
    :param v: vector involved in the constraint, `M @ x = v`
    :param C: matrix involved in the L1 regularizer
    :param c: vector involved in the L1 regularizer
    :param **kwargs: keywords passed to `lstsq_l1reg`
    :return: the solution
    """
    # check dimensions
    n = A.shape[-1]
    m = M.shape[-2]
    assert M.shape[-1] == n
    assert b.shape[0] == A.shape[-2]
    assert v.shape[0] == m
    assert m <= n
    assert C.shape[-1] == n
    assert c.shape[0] == C.shape[-2]

    Q, R = torch.linalg.qr(M.T, mode="complete")
    ystar = torch.linalg.solve_triangular(R[:m].T, v, upper=False)

    # XXX should check that there was a solution

    Atilde = A @ Q[:, m:]
    x0star = Q[:, :m] @ ystar
    btilde = b - A @ x0star

    Ctilde = C @ Q[:, m:]
    ctilde = c - C[:, :m] @ ystar
    results = lstsq_l1reg(Atilde, btilde, Ctilde, ctilde, **kwargs)
    zstar = results.solution

    xstar = x0star + Q[:, m:] @ zstar
    return xstar


def lstsq_l1reg(
    A: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    c: torch.Tensor,
    gamma: float,
    lr: float,
    max_iterations: int = 100,
) -> torch.Tensor:
    """Unconstrained L1-regularized least-squares minimization.

    Minimize `norm(A @ x - b) + gamma * l1_norm(U @ x - c)`, where the first `norm` is
    the L2 norm and the `l1_norm` is the L1 norm.

    This performs gradient descent using a coordinate system aligned with the singular
    vectors of the resulting quadratic form.

    :param A: matrix involved in minimization, `min(norm(A @ x - b))`
    :param b: vector involved in minimization, `min(norm(A @ x - b))`
    :param C: matrix involved in the L1 regularizer
    :param c: vector involved in the L1 regularizer
    :param gamma: amount of L1 regularization
    :param lr: learning rate for the gradient descent
    :param max_iterations: maximum number of iterations to run
    :return: a namespace, containing
        `solution`:     the best solution found
        `loss_history`: history of loss function values
    """
    U, S, Vh = torch.linalg.svd(A)

    m, n = A.shape
    r = min(m, n)
    if m >= n:
        Sbar_inv = torch.diag(1 / S[:r])
    else:
        Sbar_inv = torch.diag(torch.hstack((S[:r], torch.ones(n - r, dtype=S.dtype))))

    bbar = U[:, :r].T @ b
    Cbar = C @ Vh.T @ Sbar_inv

    zbar = torch.zeros((n, 1), dtype=A.dtype)
    zbar[:r] = bbar
    zbar.requires_grad_(True)
    optimizer = torch.optim.SGD([zbar], lr=lr)

    loss_history = torch.zeros(max_iterations, dtype=S.dtype)
    best_loss = None
    best_zbar = None
    for i in range(max_iterations):
        loss_sq = 0.5 * ((zbar[:r] - bbar) ** 2).sum()
        loss_l1 = gamma * torch.linalg.norm(Cbar @ zbar - c, 1)

        loss = loss_sq + loss_l1
        loss_history[i] = loss.item()

        if best_loss is None or loss.item() < best_loss:
            best_loss = loss.item()
            best_zbar = zbar.detach().clone()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    best_z = Vh.T @ Sbar_inv @ best_zbar
    return SimpleNamespace(solution=best_z, loss_history=loss_history)

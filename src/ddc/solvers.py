"""Various solvers used by the controllers."""

import torch


def lstsq_constrained(
    A: torch.Tensor, b: torch.Tensor, M: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Least-squares minimization with linear constraint.

    Minimize `norm(A @ x - b)` subject to `M @ x = v`, where the `norm` is the L2 norm.
    This assumes that `M` is full rank.

    Based on https://gist.github.com/fabianp/915461 (which in turn is based on the Golub
    & van Loan book).

    :param A: matrix involved in minimization, `min(norm(A @ x - b))`
    :param b: vector involved in minimization, `min(norm(A @ x - b))`
    :param M: matrix involved in the constraint, `M @ x = v`
    :param v: vector involved in the constraint, `M @ x = v`
    :return: the solution
    """
    # check dimensions
    n = A.shape[-1]
    m = M.shape[-2]
    assert M.shape[-1] == n
    assert b.shape[0] == A.shape[-2]
    assert v.shape[0] == m
    assert m <= n

    Q, R = torch.linalg.qr(M.T, mode="complete")
    ystar = torch.linalg.solve_triangular(R[:m].T, v, upper=False)

    # XXX should check that there was a solution

    Atilde = A @ Q[:, m:]
    x0star = Q[:, :m] @ ystar
    btilde = b - A @ x0star

    results = torch.linalg.lstsq(Atilde, btilde)
    zstar = results.solution

    # XXX should check that there was a solution
    # XXX should report whether the solution was unique?

    xstar = x0star + Q[:, m:] @ zstar
    return xstar

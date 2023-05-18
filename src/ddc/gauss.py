import torch
from scipy.linalg import sqrtm

from typing import Optional


class GaussianDistribution:
    def __init__(self, cov: torch.Tensor, mu: Optional[torch.Tensor] = None):
        """Generate samples from a multivariate normal distribution.

        :param mu: mean of the distribution; default: zero
        :param cov: covariance matrix of distribution
        """
        self.cov = cov

        assert self.cov.ndim == 2
        self.n = self.cov.shape[0]
        assert self.cov.shape[1] == self.n

        if mu is None:
            self.mu = torch.zeros(self.n, dtype=self.cov.dtype)
        else:
            self.mu = mu
            assert self.mu.ndim == 1
            assert len(self.mu) == self.n

        self.trafo = None
        self._prepare()

    def sample(
        self, count: Optional[int] = None, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Generate samples from the multivariate Gaussian.

        :param count: number of samples to generate; if not provided, generate only one
            sample; note that the dimensionality of the output is different depending on
            whether count is provided or not (see below)
        :param generator: random number generator to use; default: PyTorch default
        :return: random sample(s); if `count` is not provided, this has shape `(n,)`,
            where `n` is the dimensionality of the Guassian; if `count` is provided, the
            shape is `(count, n)` -- note that this is different from the former case
            even if `count == 1`
        """
        if count is None:
            single = True
            count = 1
        else:
            single = False

        y = torch.randn(count, self.n, generator=generator, dtype=self.trafo.dtype)
        x = y @ self.trafo

        x = x + self.mu

        if single:
            assert x.shape[0] == 1
            x = x[0]

        return x

    def _prepare(self):
        """Precalculate the transformation matrix from standard-normal to the requested
        distribution.
        """
        self.trafo = torch.tensor(sqrtm(self.cov))

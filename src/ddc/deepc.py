"""Define a DeePC implementation, according to the paper
`Data-Enabled Predictive Control: In the Shallows of the DeePC
<https://ieeexplore.ieee.org/abstract/document/8795639>`_
"""

import torch
from typing import Optional, Union, Optional, Tuple
from types import SimpleNamespace

from .solvers import lstsq_constrained


class DeepControl:
    def __init__(
        self,
        control_dim: int,
        horizon: int,
        ini_length: int = 1,
        observation_dim: Optional[int] = None,
        target: Union[float, torch.Tensor] = 0.0,
        observation_cost: float = 1.0,
        control_cost: float = 1.0,
        l2_regularization: float = 1e-3,
        control_norm_clip: Optional[float] = None,
        noise_strength: Optional[float] = None,
        relative_noise: bool = False,
        online: bool = True,
        affine: bool = False,
    ):
        self.control_dim = control_dim
        self.observation_dim = observation_dim
        self.horizon = horizon
        self.ini_length = ini_length
        self.target = target
        self.observation_cost = observation_cost
        self.control_cost = control_cost
        self.l2_regularization = l2_regularization
        self.control_norm_clip = control_norm_clip

        self.relative_noise = relative_noise
        self.online = online
        self.affine = affine

        self.seed_length = None

        if noise_strength is not None:
            self.noise_strength = noise_strength
        else:
            self.noise_strength = 0.1 if self.online else 0.0

        self.history = SimpleNamespace(
            observations=[], controls=[], controls_prenoise=[]
        )
        self.model_data = SimpleNamespace(observations=None, controls=None)

        self._latest_controls = None
        self._latest_controls_prenoise = None

    def generate_seed(
        self, n: int, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        self.seed_length = n

        # generate random direction with uniform norm distribution
        # XXX should probably do uniform density inside the unit sphere
        seed = torch.normal(torch.zeros(n, self.control_dim, dtype=dtype), 1.0)

        scaling = 1.0 if self.control_norm_clip is None else self.control_norm_clip
        seed_norm = seed / torch.linalg.norm(seed, dim=1)[:, None]
        norms = scaling * torch.rand(size=(n, 1), dtype=dtype)
        seed = seed_norm * norms

        self._latest_controls = seed
        self._latest_controls_prenoise = torch.zeros_like(seed)
        return seed

    def feed(
        self,
        observations: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        update_seed: Optional[bool] = None,
    ):
        """If controls are not provided, the most recent generated controls are used.
        If there is not enough control history available, an exception is raised.
        """
        if self.observation_dim is None:
            self.observation_dim = observations.shape[-1]

        n = len(observations)
        if controls is None:
            if len(self._latest_controls) < n:
                raise RuntimeError(
                    f"Not enough control history available, "
                    f"{len(self._latest_controls)} < {n}."
                )

            controls = self._latest_controls[:n]
            controls_prenoise = self._latest_controls_prenoise[:n]
        else:
            assert len(controls) == len(observations)
            controls_prenoise = controls

        self.history.observations.extend(observations)
        self.history.controls.extend(controls)
        self.history.controls_prenoise.extend(controls_prenoise)

        # update the seed?
        if self.model_data.observations is None:
            update_seed = True
        elif update_seed is None:
            update_seed = self.online

        if update_seed:
            k = self.seed_length
            self.update_seed(self.history.observations[-k:], self.history.controls[-k:])

    def update_seed(self, observations: torch.Tensor, controls: torch.Tensor):
        assert len(observations) == len(controls)
        self.model_data.observations = torch.stack(observations)
        self.model_data.controls = torch.stack(controls)

    def plan(self) -> torch.Tensor:
        U, Y = self.generate_hankels()
        dtype = U.dtype

        d = self.observation_dim
        c = self.control_dim

        Up = U[: c * self.ini_length]
        Yp = Y[: d * self.ini_length]

        Uf = U[c * self.ini_length :]
        Yf = Y[d * self.ini_length :]

        # constraints
        u_ini = torch.hstack(self.history.controls[-self.ini_length :])[:, None]
        y_ini = torch.hstack(self.history.observations[-self.ini_length :])[:, None]
        if self.affine:
            # affine control simply implies fixing the sum of coeffs
            e = torch.ones((1, Up.shape[1]), dtype=dtype)
            M = torch.vstack((Up, Yp, e))
            v = torch.vstack((u_ini, y_ini, torch.tensor([1.0])))
        else:
            M = torch.vstack((Up, Yp))
            v = torch.vstack((u_ini, y_ini))

        # loss function
        u_scale = self.control_cost**0.5
        y_scale = self.observation_cost**0.5
        A = torch.vstack((u_scale * Uf, y_scale * Yf))
        b = torch.zeros((len(A), 1), dtype=dtype)
        if torch.is_tensor(self.target):
            d = self.observation_dim
            k = len(Yf) // d
            assert len(Yf) % d == 0
            assert len(self.target) == d
            for i in range(k):
                b[len(Uf) + i * d : len(Uf) + (i + 1) * d] = y_scale * self.target
        else:
            b[len(Uf) :] = y_scale * self.target

        g = lstsq_constrained(A, b, M, v, self.l2_regularization)
        control_plan_prenoise = Uf @ g

        if self.control_norm_clip is not None:
            control_norms = torch.linalg.norm(control_plan_prenoise, dim=1)
            mask = control_norms > self.control_norm_clip
            control_plan_prenoise[mask, :] /= control_norms[mask, None]

        have_noise = self.noise_strength > 0
        if have_noise:
            last_observation = self.history.observations[-1]
            eps = self.noise_strength * (
                2 * torch.rand(control_plan_prenoise.shape, dtype=dtype) - 1
            )
            if self.relative_noise:
                noise = eps * torch.linalg.norm(last_observation)
            else:
                noise = eps
            control_plan = control_plan_prenoise + noise
        else:
            control_plan = control_plan_prenoise

        self._latest_controls = control_plan
        self._latest_controls_prenoise = control_plan_prenoise
        return control_plan

    def generate_hankels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = self.ini_length + self.horizon
        cols = self.seed_length + 1 - rows

        d = self.observation_dim
        c = self.control_dim

        obs = self.model_data.observations
        ctrl = self.model_data.controls

        dtype = obs.dtype
        assert ctrl.dtype == obs.dtype

        U = torch.empty((c * rows, cols), dtype=dtype)
        Y = torch.empty((d * rows, cols), dtype=dtype)

        for i in range(rows):
            U[i * d : (i + 1) * d, :] = ctrl[i : i + cols].T
            Y[i * d : (i + 1) * d, :] = obs[i : i + cols].T

        return U, Y

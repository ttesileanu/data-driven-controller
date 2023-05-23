import torch
from typing import Optional
from .gauss import GaussianDistribution


class LinearSystem:
    def __init__(
        self,
        evolution: torch.Tensor,
        control: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
        state_noise: Optional[torch.Tensor] = None,
        observation_noise: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """Create a linear dynamical system.

        The system obeys

            x[t + 1] = evolution @ x[t] + control @ u[t] + state_noise ,
            y[t] = observation @ x[t] + observation_noise ,

        where `u[t]` is the control signal and `y[t]` are the observations. Note that
        the state and observation noise are assumed independent for now.

        Note that the state can be a single column vector (the default), or a matrix. In
        the latter case each column of the state vector evolves independently -- this is
        the batch case.

        :param evolution: evolution matrix
        :param control: control matrix
        :param observation: observation matrix; default: identity
        :param state_noise: covariance matrix for state noise (assumed white); default:
            the zero matrix, i.e., no state noise
        :param observation_noise: covariance matrix for observation noise (assumed
            white); default: the zero matrix, i.e., no observation noise
        :param initial_state: value of system state at initial time, `x[0]`; default:
            zero vector
        :param generator: random number generator for state and observation noise;
            default: use PyTorch's default generator
        """
        self.evolution = evolution.detach().clone()
        self.control = control.detach().clone()

        self.state_dim = self.evolution.shape[0]
        assert self.evolution.shape[1] == self.state_dim

        self.control_dim = self.control.shape[1]
        assert self.control.shape[0] == self.state_dim

        if observation is None:
            self.observation = torch.eye(self.state_dim, dtype=self.evolution.dtype)
        else:
            self.observation = observation.detach().clone()
        self.observation_dim = self.observation.shape[0]
        assert self.observation.shape[1] == self.state_dim

        if state_noise is None:
            self.state_noise = torch.zeros(
                (self.state_dim, self.state_dim), dtype=self.evolution.dtype
            )
            self.has_state_noise = False
        else:
            self.state_noise = state_noise.detach().clone()
            self.has_state_noise = True
        assert self.state_noise.shape[0] == self.state_dim
        assert self.state_noise.shape[1] == self.state_dim

        if observation_noise is None:
            self.observation_noise = torch.zeros(
                (self.observation_dim, self.observation_dim), dtype=self.evolution.dtype
            )
            self.has_observation_noise = False
        else:
            self.observation_noise = observation_noise.detach().clone()
            self.has_observation_noise = True
        assert self.observation_noise.shape[0] == self.observation_dim
        assert self.observation_noise.shape[1] == self.observation_dim

        if initial_state is None:
            self.state = torch.zeros((self.state_dim, 1), dtype=self.evolution.dtype)
            self.batch_size = 1
        else:
            if initial_state.ndim == 1:
                initial_state = initial_state[:, None]
            self.state = initial_state.detach().clone()
            self.batch_size = self.state.shape[1]
        assert self.state.shape[0] == self.state_dim

        self.generator = generator

        self._sanitize_types()

        if self.has_state_noise:
            self._state_gaussian = GaussianDistribution(self.state_noise)
        if self.has_observation_noise:
            self._observation_gaussian = GaussianDistribution(self.observation_noise)

    def run(
        self,
        n_steps: Optional[int] = None,
        control_plan: Optional[torch.Tensor] = None,
        store_initial: bool = False,
    ) -> torch.Tensor:
        """Run the dynamical system for a number of steps.

        States are stored by default *after* the dynamics acts, but see `store_initial`
        for storing the initial state, as well (which is equivalent to storing the state
        before the dynamics, except that the final state is also appended to the return
        tensor).

        If the number of steps is provided without a `control_plan`, the function will
        simulate an autonomous system (i.e., with control set to 0).

        If a `control_plan` is provided, it will set the control inputs received by the
        system. Generally the number of steps can be inferred from the shape of the
        `control_plan` (see below); if `n_steps` is also provided, it should be
        compatible with the shape of the `control_plan`.

        :param n_step: number of steps to simulate; if not provided, infer from
            `control_plan` (see above)
        :param control_plan: control inputs to be provided to the system; should have
            shape `(n, control_dim)` or `(n, control_dim, batch_size); in the former
            case, if `batch_size > 1`, the same control is used for all samples
        :param store_initial: whether to store the initial state of the system, before
            any dynamics steps, or only store the subsequent evolution
        :return: the system observations, with shape `(n_out, observation_dim,
            batch_size)`, where `n_out` can be `n` (if `store_initial` is false) or
            `n + 1` (if `store_initial` is true)
        """
        if control_plan is not None:
            assert control_plan.ndim in [2, 3]
            if n_steps is not None:
                assert n_steps == control_plan.shape[0]
            else:
                n_steps = control_plan.shape[0]

            if control_plan.ndim < 3:
                control_plan = torch.tile(control_plan[..., None], (self.batch_size,))
            assert control_plan.ndim == 3
            assert control_plan.shape[-1] == self.batch_size

            assert control_plan.shape[1] == self.control_dim

        n_out = n_steps + store_initial
        observations = torch.empty(
            n_out, self.observation_dim, self.batch_size, dtype=self.state.dtype
        )
        if store_initial:
            observations[0] = self.observe()
        for i in range(n_steps):
            # evolve
            x = self.evolution @ self.state
            if control_plan is not None:
                x += self.control @ control_plan[i]
            if self.has_state_noise:
                x += self._state_gaussian.sample(
                    self.batch_size, generator=self.generator
                ).T

            self.state = x

            # observe
            observations[i + store_initial] = self.observe()

        return observations

    def observe(self) -> torch.Tensor:
        """Read out one observation (or one batch of observations) from the system.

        Note that calling `observe` multiple times on the same state can yield different
        results if `observation_noise` is non-zero.

        :return: system observation(s), shape `(observation_dim, batch_size)`
        """
        y = self.observation @ self.state
        if self.has_observation_noise:
            y += self._observation_gaussian.sample(
                self.batch_size, generator=self.generator
            ).T
        return y

    def convert_type(self, dtype: torch.dtype) -> "LinearSystem":
        """Convert all model tensors to the given type.

        :return: self
        """
        self.evolution = self.evolution.type(dtype)
        self.control = self.control.type(dtype)
        self.observation = self.observation.type(dtype)
        self.state = self.state.type(dtype)

        if self.has_state_noise:
            self.state_noise = self.state_noise.type(dtype)
        if self.has_observation_noise:
            self.observation_noise = self.observation_noise.type(dtype)

        return self

    def _sanitize_types(self):
        """Promote all tensors to a common type."""
        # find the common type
        common_type = self.evolution.dtype
        for t in [self.control, self.observation, self.state]:
            common_type = torch.promote_types(common_type, t.dtype)

        if self.has_state_noise:
            common_type = torch.promote_types(common_type, self.state_noise.dtype)
        if self.has_observation_noise:
            common_type = torch.promote_types(common_type, self.observation_noise.dtype)

        # promote
        self.convert_type(common_type)

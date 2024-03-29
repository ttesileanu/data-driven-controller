import torch
from typing import Optional, Callable
from .gauss import GaussianDistribution


class GeneralSystem:
    def __init__(
        self,
        evolution: Callable,
        observation: Callable = None,
        state_dim: Optional[int] = None,
        control_dim: Optional[int] = None,
        state_noise: Optional[torch.Tensor] = None,
        observation_noise: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Create a linear dynamical system.

        The system obeys

            x[t + 1] = evolution(x[t], u[t]) + state_noise ,
            y[t] = observation(x[t]) + observation_noise ,

        where `u[t]` is the control signal and `y[t]` are the observations. Note that
        the state and observation noise are assumed independent and additive.

        Note that the state can be a single column vector (the default), or a matrix. In
        the latter case each column of the state vector evolves independently -- this is
        the batch case.

        :param evolution: evolution function, dependent on current state and control
        :param observation: observation function; default: identity
        :param state_dim: dimensionality of state; inferred from `state_noise` or
            `initial_state` if given
        :param control_dim: dimensionality of control; if not given, it will be inferred
            from the first call to `run` that has a `control_plan`
        :param state_noise: covariance matrix for state noise (assumed white); default:
            the zero matrix, i.e., no state noise
        :param observation_noise: covariance matrix for observation noise (assumed
            white); default: the zero matrix, i.e., no observation noise
        :param initial_state: value of system state at initial time, `x[0]`; default:
            zero vector, but needs a way to infer dimensionality; se `state_dim`
        :param generator: random number generator for state and observation noise;
            default: use PyTorch's default generator
        """
        self.evolution = evolution

        self.state_dim = state_dim
        if state_noise is not None:
            if self.state_dim is None:
                self.state_dim = len(state_noise)
            else:
                assert self.state_dim == len(state_noise)

        if observation is None:
            self.observation = lambda x: x
        else:
            self.observation = observation

        self.control_dim = control_dim

        if state_noise is None:
            self.state_noise = None
            self.has_state_noise = False
        else:
            self.state_noise = state_noise.detach().clone()
            self.has_state_noise = True

            if self.state_dim is not None:
                assert self.state_noise.shape[0] == self.state_dim
            else:
                self.state_dim = self.state_noise.shape[0]

            assert self.state_noise.shape[1] == self.state_dim

        if observation_noise is None:
            self.observation_noise = None
            self.has_observation_noise = False
            self.observation_dim = None
        else:
            self.observation_noise = observation_noise.detach().clone()
            self.has_observation_noise = True

            self.observation_dim = self.observation_noise.shape[0]
            assert self.observation_noise.shape[1] == self.observation_dim

        self.dtype = dtype
        if initial_state is None:
            assert self.state_dim is not None
            self.state = torch.zeros((self.state_dim, 1), dtype=self.dtype)
            self.batch_size = 1
        else:
            if initial_state.ndim == 1:
                initial_state = initial_state[:, None]
            self.state = initial_state.detach().clone()
            self.batch_size = self.state.shape[1]

            if self.state_dim is not None:
                assert self.state.shape[0] == self.state_dim
            else:
                self.state_dim = len(initial_state)

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
        store_initial: bool = True,
        store_final: bool = False,
    ) -> torch.Tensor:
        """Run the dynamical system for a number of steps.

        States are stored by default *after* the dynamics acts, but see `store_initial`
        for storing the initial state, as well (which is equivalent to storing the state
        before the dynamics, except that the final state is also appended to the return
        tensor).

        If the number of steps is provided without a `control_plan`, the function will
        simulate an autonomous system (i.e., with control set to 0). Specifically, if
        `self.control_dim` is set, the evolution function will be given a zero tensor of
        size `(self.control_dim, batch_size)` for the control. Otherwise, a floating
        point `0.0` will be passed, and the evolution function will need to be able to
        handle this.

        If a `control_plan` is provided, it will set the control inputs received by the
        system. Generally the number of steps can be inferred from the shape of the
        `control_plan` (see below); if `n_steps` is also provided, it should be
        compatible with the shape of the `control_plan`.

        :param n_step: number of steps to simulate; if not provided, infer from
            `control_plan` (see above)
        :param control_plan: control inputs to be provided to the system; should have
            shape `(n, control_dim)` or `(n, control_dim, batch_size); in the former
            case, if `batch_size > 1`, the same control is used for all samples
        :param store_initial: whether to store the initial observation from the system,
            before any dynamics steps, or only store the subsequent evolution
        :param store_final: whether to store a final observation from the system, after
            all the dynamics steps
        :return: the system observations, with shape `(n_out, observation_dim,
            batch_size)`, where `n_out` can be `n` (if `store_initial` or `store_final`,
            but not both, are false); `n + 1` (if both `store_initial` and `store_final`
            are true); or `n - 1` (if both `store_initial` and `store_final` are false)
        """
        output_batch = True
        if control_plan is not None:
            assert control_plan.ndim in [2, 3]
            if n_steps is not None:
                assert n_steps == control_plan.shape[0]
            else:
                n_steps = control_plan.shape[0]

            if control_plan.ndim < 3:
                output_batch = self.batch_size != 1
                control_plan = torch.tile(control_plan[..., None], (self.batch_size,))
            assert control_plan.ndim == 3
            assert control_plan.shape[-1] == self.batch_size

            if self.control_dim is not None:
                assert control_plan.shape[1] == self.control_dim
            else:
                self.control_dim = control_plan.shape[1]

        n_out = n_steps - 1 + store_initial + store_final

        observations = None
        if store_initial:
            observation = self.observe()
            if observations is None:
                # dimension might just be learned from call to `self.observe()`
                observations = torch.empty(
                    n_out, self.observation_dim, self.batch_size, dtype=self.dtype
                )
            observations[0] = observation

        for i in range(n_steps):
            # evolve
            if control_plan is not None:
                action = control_plan[i]
            else:
                if self.control_dim is not None:
                    action = torch.zeros((self.control_dim, self.batch_size))
                else:
                    action = 0.0
            x = self.evolution(self.state, action)
            assert len(x) == self.state_dim

            if self.has_state_noise:
                x += self._state_gaussian.sample(
                    self.batch_size, generator=self.generator
                ).T

            self.state = x

            # observe
            if i < n_steps - 1 or store_final:
                observation = self.observe()
                if observations is None:
                    observations = torch.empty(
                        n_out, self.observation_dim, self.batch_size, dtype=self.dtype
                    )
                observations[i + store_initial] = observation

        if not output_batch:
            assert observations.shape[-1] == 1
            observations = observations[..., 0]
        return observations

    def observe(self) -> torch.Tensor:
        """Read out one observation (or one batch of observations) from the system.

        Note that calling `observe` multiple times on the same state can yield different
        results if `observation_noise` is non-zero.

        :return: system observation(s), shape `(observation_dim, batch_size)`
        """
        y = self.observation(self.state)
        if self.observation_dim is None:
            self.observation_dim = len(y)
        else:
            assert len(y) == self.observation_dim

        if self.has_observation_noise:
            y += self._observation_gaussian.sample(
                self.batch_size, generator=self.generator
            ).T
        return y

    def convert_type(self, dtype: torch.dtype) -> "GeneralSystem":
        """Convert all model tensors to the given type.

        :return: self
        """
        self.state = self.state.type(dtype)

        if self.has_state_noise:
            self.state_noise = self.state_noise.type(dtype)
        if self.has_observation_noise:
            self.observation_noise = self.observation_noise.type(dtype)

        self.dtype = dtype

        return self

    def _sanitize_types(self):
        """Promote all tensors to a common type."""
        # try to find a common type
        if self.dtype is not None:
            dtype = self.dtype
        else:
            dtype = self.state.dtype

            if self.has_state_noise:
                dtype = torch.promote_types(dtype, self.state_noise.dtype)
            if self.has_observation_noise:
                dtype = torch.promote_types(dtype, self.observation_noise.dtype)

        # promote
        self.convert_type(dtype)


class AffineControlSystem(GeneralSystem):
    def __init__(
        self,
        evolution: Callable,
        control: Callable,
        observation: Callable = None,
        **kwargs
    ):
        evolution_combined = lambda x, u: evolution(x) + control(u)
        super().__init__(evolution_combined, observation, **kwargs)

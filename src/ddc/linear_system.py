import torch
from typing import Optional

from .general_system import GeneralSystem


class LinearSystem(GeneralSystem):
    def __init__(
        self,
        evolution: torch.Tensor,
        control: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
        **kwargs,
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
        :param **kwargs: other keyword arguments passed to `GeneralSystem`
        """
        evolution_fct = lambda x: evolution @ x
        control_fct = lambda u: control @ u
        observation_fct = None if observation is None else lambda x: observation @ x

        state_dim = len(evolution)
        kwargs.setdefault("state_dim", state_dim)

        assert evolution.shape[1] == state_dim
        assert control.shape[0] == state_dim

        if observation is not None:
            assert observation.shape[1] == state_dim

        super().__init__(evolution_fct, control_fct, observation_fct, **kwargs)

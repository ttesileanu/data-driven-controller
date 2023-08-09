import torch
from typing import Optional

from ddc.general_system import GeneralSystem

from .general_system import AffineControlSystem


class LinearSystem(AffineControlSystem):
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
        self.evolution_matrix = evolution
        self.control_matrix = control

        evolution_fct = lambda x: self.evolution_matrix @ x
        control_fct = lambda u: self.control_matrix @ u

        if observation is not None:
            self.observation_matrix = observation
            observation_fct = lambda x: self.observation_matrix @ x
        else:
            self.observation_matrix = torch.eye(len(evolution), dtype=evolution.dtype)
            observation_fct = None

        state_dim = len(evolution)
        control_dim = control.shape[1]
        kwargs.setdefault("state_dim", state_dim)
        kwargs.setdefault("control_dim", control_dim)

        assert evolution.shape[1] == state_dim
        assert control.shape[0] == state_dim

        if observation is not None:
            assert observation.shape[1] == state_dim

        super().__init__(evolution_fct, control_fct, observation_fct, **kwargs)

    def convert_type(self, dtype: torch.dtype):
        super().convert_type(dtype)

        self.evolution_matrix = self.evolution_matrix.type(dtype)
        self.control_matrix = self.control_matrix.type(dtype)
        self.observation_matrix = self.observation_matrix.type(dtype)

        return self

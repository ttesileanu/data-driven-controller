import torch


class DDController:
    def __init__(
        self,
        observation_dim: int,
        control_dim: int,
        history_length: int,
        seed_length: int = 1,
        control_horizon: int = 1,
        observation_match_cost: float = 1.0,
        control_match_cost: float = 1.0,
        target_cost: float = 0.01,
        control_cost: float = 0.01,
        control_sparsity: float = 0.0,
    ):
        """Data-drive controller.

        This acts by leveraging the behavioral approach to dynamical systems: a system
        is identified by the fact that its possible trajectories are restricted to a
        certain subspace. Observing actual trajectories can then be used to identify
        this subspace and, thus, the system.

        Given a matrix `Y` in which each column is a sequence of observations of a
        dynamical system and a matrix `U` in which each column is a sequence of control
        inputs to the system, and assuming that the number of columns is large enough,
        any new sequence of controls `u` and observations `y` can be written as a linear
        combination of the earlier data:

            z = Z alpha ,

        where `Z = vstack((Y, U))`, `z = vstack((y, u))`, and `alpha` is a vector of
        coefficients. The matrices `Y` and `U` can be obtained from a single long
        sequence in the form of Hankel matrices -- and that is what we will be doing
        here.

        In our case, we want to establish a control plan that maximally reduces the
        magnitude of the observations after a certain horizon, using the earlier
        measurements to predict the behavior of the system. To perform this prediction
        we need to start with a "seed" where the observations and control are known --
        the seed should be sufficiently long to fully specify the (unobserved) internal
        state of the system.

        More specifically, suppose the system's state is fully identified by observing
        `seed_length` steps and suppose that we're trying to minimize the system's
        output a number `control_horizon` of steps in the future. Since we want to make
        sure that the full (unobserved) state of the system is under control, we also
        evaluate the target output by summing over `seed_length` steps. The number of
        prior samples used in the prediction is given by `history_length`.

        The control planning is set up as an optimization problem where the coefficients
        `alpha` are inferred such that a loss function containing five terms is
        minimized:

            L = (observation_match_cost * (predicted - measured seed observations)) ** 2
              + (control_match_cost * (predicted - measured seed controls)) ** 2
              + (target_cost * (predicted observations at horizon)) ** 2
              + (control_cost * (predicted control after seed)) ** 2
              + control_sparsity * L1_norm(predicted control after seed) .

        The inferred coefficients are then used to predict the optimal control. Note
        that apart from the L1 regularization term, this is just a least-squares
        problem.

        :param observation_dim: dimensionality of measurements
        :param control_dim: dimensionality of control
        :param history_length: number of columns to use in the Hankel matrices
        :param seed_length: number of measurements needed to fully specify an internal
            state
        :param control_horizon: how far ahead to aim for reduced measurements
        :param observation_match_cost: multiplier for matching observation values
        :param control_match_cost: multiplier for matching seed control values
        :param target_cost: multiplier for minimizing observation values at horizon
        :param control_cost: multiplier for minimizing overall control magnitude
        :param control_sparsity: amount of sparsity-inducing, L1 regularizer; NOT YET
            IMPLEMENTED
        """
        self.observation_dim = observation_dim
        self.control_dim = control_dim
        self.history_length = history_length
        self.seed_length = seed_length
        self.control_horizon = control_horizon
        self.observation_match_cost = observation_match_cost
        self.control_match_cost = control_match_cost
        self.target_cost = target_cost
        self.control_cost = control_cost
        self.control_sparsity = control_sparsity
        if self.control_sparsity != 0:
            assert NotImplementedError("control_sparsity not implemented yet")

        self.observation_history = None
        self.control_history = None

    def feed(self, observation: torch.Tensor, control: torch.Tensor):
        """Register another measurement.

        At most `self.history_length` samples are kept.

        :param observation: output from the model
        :param control: control input
        """
        assert len(observation) == self.observation_dim
        assert len(control) == self.control_dim

        if self.observation_history is None:
            self.observation_history = observation.clone()
        else:
            self.observation_history = torch.vstack(
                (self.observation_history, observation)
            )
        if self.control_history is None:
            self.control_history = control.clone()
        else:
            self.control_history = torch.vstack((self.control_history, control))

        n = self.history_length + self.control_horizon + self.seed_length
        self.observation_history = self.observation_history[-n:]
        self.control_history = self.control_history[-n:]

    def plan(self) -> torch.Tensor:
        """Estimate optimal control plan given the current history.

        :return: control plan, shape `(control_horizon, control_dim)`
        """
        p = self.seed_length
        l = self.control_horizon
        n = self.history_length

        d = self.observation_dim
        c = self.control_dim

        if len(self.observation_history) < n + p + l:
            # not enough data yet
            return torch.zeros((l, c))

        obs = self.observation_history
        ctrl = self.control_history

        end = len(obs)
        assert len(ctrl) == end

        Y = torch.empty((2 * p * d, n))
        U = torch.empty(((p + l) * c, n))
        # target y and target z are zero
        y = torch.zeros((2 * p * d, 1))
        u = torch.zeros((p + l) * c, 1)
        for i in range(p):
            yi = i * d
            Y[yi : yi + d] = obs[i : i + n].T
            y[yi : yi + d] = obs[end + i - p][:, None]

            yi = (i + p) * d
            Y[yi : yi + d] = obs[l + i : l + i + n].T

            ui = i * c
            U[ui : ui + c] = ctrl[i : i + n].T
            u[ui : ui + c] = ctrl[end + i - p][:, None]

        for i in range(p, p + l):
            ui = i * c
            U[ui : ui + c] = ctrl[i : i + n].T

        Z = torch.vstack((Y, U))
        z = torch.vstack((y, u))

        # weigh using the appropriate coefficients
        Z_weighted = Z.clone()
        Z_weighted[: p * d] *= self.observation_match_cost
        Z_weighted[p * d : 2 * p * d] *= self.target_cost
        Z_weighted[2 * p * d : p * (2 * d + c)] *= self.control_match_cost
        Z_weighted[-l * c :] *= self.control_cost

        z_weighted = z.clone()
        z_weighted[: p * d] *= self.observation_match_cost
        # nothing to multiply for [p * d : 2 * p * d] as target y is zero

        z_weighted[2 * p * d : p * (2 * d + c)] *= self.control_match_cost
        # nothing to multiply for [-l * c :] as target u is zero

        # now solve z = Z alpha
        result = torch.linalg.lstsq(Z_weighted, z_weighted)
        z_hat = Z @ result.solution

        return z_hat[-l * c :].reshape((l, c))

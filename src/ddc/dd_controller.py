import torch

from typing import Tuple

from .solvers import lstsq_constrained, lstsq_l1reg_constrained


class DDController:
    def __init__(
        self,
        observation_dim: int,
        control_dim: int,
        history_length: int,
        seed_length: int = 1,
        control_horizon: int = 1,
        output_cost: float = 0.01,
        target_cost: float = 0.01,
        control_cost: float = 0.01,
        control_sparsity: float = 0.0,
        method: str = "lstsq",
        gd_lr: float = 0.01,
        gd_iterations: int = 50,
        noise_handling: str = "none",
        eager_start: bool = False,
        offline: bool = False,
    ):
        """Data-driven controller.

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
        `alpha` are inferred such that the following loss function is minimized, subject
        to a constraint:

            L = (output_cost * (predicted observations between seed and horizon)) ** 2
              + (target_cost * (predicted observations at horizon)) ** 2
              + (control_cost * (predicted control after seed)) ** 2
              + control_sparsity * L1_norm(predicted control after seed) ,

        subject to

            predicted - measured seed observations = 0 , and
            predicted - measured seed controls = 0 .

        The inferred coefficients are then used to predict the optimal control. Note
        that apart from the L1 regularization term, this is just a least-squares
        problem with a linear constraint.

        By default the controller assumes that the dynamics is noiseless. If there is
        noise, the default controller will be suboptimal -- it will basically treat
        noise optimistically and assume that it will act in favor of stabilization. To
        treat noise more appropriately, set `noise_handling` to `"average"`. This
        averages over sets of columns of the Hankel matrices until the linear system
        that is solved has the number of equations plus the number of constraints equal
        to the number of unknowns. Specifically, the number of columns is set to

            n_columns = (control_horizon - seed_length - 1) * control_dim ,

        and if `history_length > n_columns`, averaging is done until the number of
        columns drops down to `n_columns`. (An exception is raised if `history_length`
        is smaller than `n_columns`.)

        :param observation_dim: dimensionality of measurements
        :param control_dim: dimensionality of control
        :param history_length: number of columns to use in the Hankel matrices
        :param seed_length: number of measurements needed to fully specify an internal
            state
        :param control_horizon: how far ahead to aim for reduced measurements
        :param output_cost: multiplier for minimizing observation values after seed but
            before target
        :param target_cost: multiplier for minimizing observation values at horizon
        :param control_cost: multiplier for minimizing overall control magnitude
        :param control_sparsity: amount of sparsity-inducing, L1 regularizer; only works
            when `method == "gd"`
        :param method: can be
            "lstsq":    use `torch.linalg.lstsq`; does not support `control_sparsity`
            "gd":       gradient descent
        :param gd_lr: learning rate for `method == "gd"`
        :param gd_iterations: number of iterations when `method == "gd"`
        :param noise_handling: method for handling noisy data; can be
            "none":     assume noiseless observations
            "average":  average over samples; see above
        :param eager_start: if true, return non-trivial controls as soon as the minimal
            number of samples required for a solution are available, even if the
            requested `history_length` has not been reached
        :param offline: if true, the model freezes itself as soon as it collected enough
            data
        """
        self.observation_dim = observation_dim
        self.control_dim = control_dim
        self.history_length = history_length
        self.seed_length = seed_length
        self.control_horizon = control_horizon
        self.output_cost = output_cost
        self.target_cost = target_cost
        self.control_cost = control_cost
        self.control_sparsity = control_sparsity
        self.method = method
        self.gd_lr = gd_lr
        self.gd_iterations = gd_iterations
        self.noise_handling = noise_handling
        self.eager_start = eager_start

        if self.control_sparsity != 0:
            if self.method != "gd":
                raise ValueError("control_sparsity only works with `method=gd`")

        self.observation_history = None
        self.control_history = None

        self.frozen_observation_history = None
        self.frozen_control_history = None

        self.offline = offline
        self.frozen = False

        self._previous_coeffs = None

    def feed(self, control: torch.Tensor, observation: torch.Tensor):
        """Register another measurement.

        The necessary number of samples is kept so that the Hankel matrices can have
        `self.history_length` columns.

        The very first recorded control value is never used, so the history of control
        values always has one element less than the history of observations.

        :param control: control input *preceding* the latest observation (output)
        :param observation: latest output from the model
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
        assert n > 1

        self.observation_history = self.observation_history[-n:]

        n_ctrl = len(self.observation_history) - 1
        self.control_history = self.control_history[-n_ctrl:]

    def plan(self) -> torch.Tensor:
        """Estimate optimal control plan given the current history.

        :return: control plan, shape `(control_horizon, control_dim)`
        """
        p = self.seed_length
        l = self.control_horizon
        c = self.control_dim
        d = self.observation_dim
        if len(self.observation_history) < self.minimal_history:
            # not enough data yet
            return torch.zeros((l, c), dtype=self.observation_history.dtype)
        elif self.offline:
            self.freeze()

        # Z, _, Z_weighted, z_weighted = self.get_hankels()
        Zm, Zp, Zm_weighted, Zp_weighted = self.get_hankels()
        zm, zp, zm_weighted, zp_weighted = self.get_targets()

        if self.method == "lstsq":
            coeffs = lstsq_constrained(
                Zp_weighted, zp_weighted, Zm_weighted, zm_weighted
            )
        elif self.method == "gd":
            U_unk = Zp[-l * c :]
            coeffs = lstsq_l1reg_constrained(
                Zp_weighted,
                zp_weighted,
                Zm_weighted,
                zm_weighted,
                U_unk,
                torch.zeros((l * c, 1), dtype=self.observation_history.dtype),
                gamma=self.control_sparsity,
                lr=self.gd_lr,
                max_iterations=self.gd_iterations,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # now solve z = Z alpha
        z_hat = Zp @ coeffs

        return z_hat[-l * c :].reshape((l, c))

    def get_hankels(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate Hankel matrices.

        :return: tuple `(Zm, Zp, Zm_weighted, Zp_weighted)`, where `m` stands for match
            (in the seed region) and `p` stands for predict; weighted values are
            weighted by the relevant costs
        """
        p = self.seed_length
        l = self.control_horizon

        d = self.observation_dim
        c = self.control_dim

        obs, ctrl = self.get_past_trajectories()

        end = len(obs)
        assert len(ctrl) == end - 1

        n = end - l - p

        if self.output_cost == 0:
            ydim = 2 * p * d
        else:
            ydim = (p + l) * d
        # we need one fewer controls than observations
        udim = (p + l - 1) * c
        zdim = ydim + udim
        dtype = obs.dtype
        Z = torch.empty((zdim, n), dtype=dtype)
        # target y and target u are zero; we update the 'match' components below
        z = torch.zeros((zdim, 1), dtype=dtype)

        matchdim = p * d + (p - 1) * c
        for i in range(p):
            yi = i * d
            Z[yi : yi + d] = obs[i : i + n].T

            if i < p - 1:
                # we need one fewer controls than observations in the match region
                ui = p * d + i * c
                Z[ui : ui + c] = ctrl[i : i + n].T

        if self.control_cost == 0:
            for i in range(p):
                yi = matchdim + i * d
                Z[yi : yi + d] = obs[l + i : l + i + n].T
        else:
            for i in range(p, l + p):
                yi = matchdim + (i - p) * d
                Z[yi : yi + d] = obs[i : i + n].T

        if self.output_cost == 0:
            startui = matchdim + p * d
        else:
            startui = matchdim + l * d
        for i in range(p - 1, p + l - 1):
            ui = startui + (i - p + 1) * c
            Z[ui : ui + c] = ctrl[i : i + n].T

        # average if needed
        if self.noise_handling == "average":
            n_columns = p * d + l * c
            if n < n_columns:
                raise ValueError(f"history too short for noise averaging.")
            bins = torch.linspace(0, n, n_columns + 1, dtype=int)

            Z0 = Z
            Z = torch.empty((zdim, n_columns), dtype=dtype)
            for i, (i0, i1) in enumerate(zip(bins, bins[1:])):
                Z[:, i] = Z0[:, i0:i1].mean(dim=1)
        elif self.noise_handling != "none":
            raise ValueError(f"Unknown noise handling method: {self.noise_handling}.")

        # weigh using the appropriate coefficients
        Z_weighted = Z.clone()

        if self.output_cost != 0:
            Z_weighted[matchdim : startui - p * d] *= self.output_cost
        Z_weighted[startui - p * d : startui] *= self.target_cost
        Z_weighted[-l * c :] *= self.control_cost

        Zm = Z[:matchdim]
        Zp = Z[matchdim:]
        Zm_weighted = Z_weighted[:matchdim]
        Zp_weighted = Z_weighted[matchdim:]
        return Zm, Zp, Zm_weighted, Zp_weighted

    def get_past_trajectories(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get matrices of past observations and control.

        These can be obtained either online (from the newest set of measurements) or
        offline (from the time `self.freeze()` was called).

        :return: tuple `(obs, ctrl)` of observations and controls
        """
        if not self.frozen:
            return self.observation_history, self.control_history
        else:
            return self.frozen_observation_history, self.frozen_control_history

    def get_targets(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate target values.

        :return: tuple `(zm, zp, zm_weighted, zp_weighted)`, where `m` stands for match
            (in the seed region) and `p` stands for predict; weighted values are
            weighted by the relevant costs
        """
        p = self.seed_length
        l = self.control_horizon

        d = self.observation_dim
        c = self.control_dim

        obs = self.observation_history
        ctrl = self.control_history

        end = len(obs)
        assert len(ctrl) == end - 1

        n = end - l - p

        if self.output_cost == 0:
            ydim = 2 * p * d
        else:
            ydim = (p + l) * d
        # we need one fewer controls than observations
        udim = (p + l - 1) * c
        zdim = ydim + udim
        dtype = obs.dtype
        # target y and target u are zero; we update the 'match' components below
        z = torch.zeros((zdim, 1), dtype=dtype)

        matchdim = p * d + (p - 1) * c
        for i in range(p):
            yi = i * d
            z[yi : yi + d] = obs[end + i - p][:, None]

            if i < p - 1:
                # we need one fewer controls than observations in the match region
                ui = p * d + i * c
                z[ui : ui + c] = ctrl[end + i - p][:, None]

        # weigh using the appropriate coefficients
        # z_weighted = z.clone()
        z_weighted = z

        # return Z, z, Z_weighted, z_weighted
        Zm = z[:matchdim]
        Zp = z[matchdim:]
        Zm_weighted = z_weighted[:matchdim]
        Zp_weighted = z_weighted[matchdim:]
        return Zm, Zp, Zm_weighted, Zp_weighted

    def freeze(self):
        """Freeze the model used by the controller."""
        if not self.frozen:
            self.frozen_control_history = self.control_history.clone()
            self.frozen_observation_history = self.observation_history.clone()

            self.frozen = True

    def unfreeze(self):
        """Unfreeze the model used by the controller."""
        if self.frozen:
            self.frozen_control_history = None
            self.frozen_observation_history = None

            self.frozen = False

    @property
    def minimal_history(self):
        """The minimal number of steps needed to calculate non-trivial control."""
        p = self.seed_length
        l = self.control_horizon
        d = self.observation_dim
        c = self.control_dim

        if self.eager_start:
            if self.output_cost == 0:
                # we have a total dimension
                #   zdim = 2 * p * d + (p + l - 1) * c
                # and a number of constraints
                #   matchdim = p * d + (p - 1) * c
                # the number of non-constraint rows is
                #   zdim - matchdim = p * d + l * c
                # and the number of degrees of freedom is
                #   zdim - 2 * matchdim = (l - p - 1) * c

                # end = len(obs) must equal l + p + n
                # for rank reasons, I want n = (l - p - 1) * c
                # end = l + p + (l - p - 1) * c
                return l + p + (l - p - 1) * c
            else:
                # we have a total dimension
                #   zdim = (p + l) * (d + c) - c
                # and a number of constraints
                #   matchdim = p * d + (p - 1) * c
                # the number of non-constraint rows is
                #   zdim - matchdim = l * (d + c)
                # and the number of degrees of freedom is
                #   zdim - 2 * matchdim = (l - p) * (d + c) + c

                # end = len(obs) must equal l + p + n
                # for rank reasons, I want n = (l - p) * (d + c) + c
                # end = l + p + c + (l - p) * (d + c)
                return l + p + c + (l - p) * (d + c)
        else:
            return self.history_length + p + l

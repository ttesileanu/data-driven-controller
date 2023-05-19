import torch

from typing import Tuple

from .solvers import lstsq_constrained


class DDController:
    def __init__(
        self,
        observation_dim: int,
        control_dim: int,
        history_length: int,
        seed_length: int = 1,
        control_horizon: int = 1,
        exact_match: bool = True,
        observation_match_cost: float = 1.0,
        control_match_cost: float = 1.0,
        target_cost: float = 0.01,
        control_cost: float = 0.01,
        control_sparsity: float = 0.0,
        method: str = "lstsq",
        gd_lr: float = 0.01,
        gd_iterations: int = 50,
        noise_handling: str = "none",
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

            L = (target_cost * (predicted observations at horizon)) ** 2
              + (control_cost * (predicted control after seed)) ** 2
              + control_sparsity * L1_norm(predicted control after seed) ,

        subject to

            predicted - measured seed observations = 0 , and
            predicted - measured seed controls = 0 .

        The inferred coefficients are then used to predict the optimal control. Note
        that apart from the L1 regularization term, this is just a least-squares
        problem with a linear constraint.

        If `exact_match` is set to false, the constraint is treated softly by
        incorporating it into the objective:

            L = (observation_match_cost * (predicted - measured seed observations)) ** 2
              + (control_match_cost * (predicted - measured seed controls)) ** 2
              + (target_cost * (predicted observations at horizon)) ** 2
              + (control_cost * (predicted control after seed)) ** 2
              + control_sparsity * L1_norm(predicted control after seed) .

        By default the controller assumes that the dynamics is noiseless. If there is
        noise, the default controller will be suboptimal -- it will basically treat
        noise optimistically and assume that it will act in favor of stabilization. To
        treat noise more appropriately, set `noise_handling` to `"average"`. This
        averages over sets of columns of the Hankel matrices until the linear system
        that is solved has the number of equations plus the number of constraints equal
        to the number of unknowns. Specifically, the number of columns is set to

            n_columns = (control_horizon - seed_length) * control_dim ,

        and if `history_length > n_columns`, averaging is done until the number of
        columns drops down to `n_columns`. (An exception is raised if `history_length`
        is smaller than `n_columns`.)

        :param observation_dim: dimensionality of measurements
        :param control_dim: dimensionality of control
        :param history_length: number of columns to use in the Hankel matrices
        :param seed_length: number of measurements needed to fully specify an internal
            state
        :param control_horizon: how far ahead to aim for reduced measurements
        :param exact_match: whether the observations and controls during the seed period
            are matched exactly or as a soft constraint; see above
        :param observation_match_cost: multiplier for matching observation values; not
            used if `exact_match == True`
        :param control_match_cost: multiplier for matching seed control values; not used
            if `exact_match == True`
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
        """
        self.observation_dim = observation_dim
        self.control_dim = control_dim
        self.history_length = history_length
        self.seed_length = seed_length
        self.control_horizon = control_horizon
        self.exact_match = exact_match
        self.observation_match_cost = observation_match_cost
        self.control_match_cost = control_match_cost
        self.target_cost = target_cost
        self.control_cost = control_cost
        self.control_sparsity = control_sparsity
        self.method = method
        self.gd_lr = gd_lr
        self.gd_iterations = gd_iterations
        self.noise_handling = noise_handling

        if self.control_sparsity != 0:
            if self.method != "gd":
                assert ValueError("control_sparsity only works with `method=gd`")
            if exact_match:
                assert NotImplementedError(
                    "exact_match not implemented for `method=gd`"
                )

        self.observation_history = None
        self.control_history = None

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
        self.control_history = self.control_history[-(n - 1) :]

    def plan(self) -> torch.Tensor:
        """Estimate optimal control plan given the current history.

        :return: control plan, shape `(control_horizon, control_dim)`
        """
        n = self.history_length
        p = self.seed_length
        l = self.control_horizon
        c = self.control_dim
        if len(self.observation_history) < n + p + l:
            # not enough data yet
            return torch.zeros((l, c))

        Z, _, Z_weighted, z_weighted = self.get_hankels()

        if self.method == "lstsq":
            if not self.exact_match:
                result = torch.linalg.lstsq(Z_weighted, z_weighted)
                coeffs = result.solution
            else:
                d = self.observation_dim
                matchdim = p * d + (p - 1) * c
                coeffs = lstsq_constrained(
                    Z_weighted[matchdim:],
                    z_weighted[matchdim:],
                    Z_weighted[:matchdim],
                    z_weighted[:matchdim],
                )
        elif self.method == "gd":
            U_unk = Z[-l * c :]
            coeffs = self._solve_gd(Z_weighted, z_weighted, U_unk)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # now solve z = Z alpha
        z_hat = Z @ coeffs

        return z_hat[-l * c :].reshape((l, c))

    def get_hankels(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.seed_length
        l = self.control_horizon
        n = self.history_length

        d = self.observation_dim
        c = self.control_dim

        obs = self.observation_history
        ctrl = self.control_history

        end = len(obs)
        assert len(ctrl) == end - 1

        ydim = 2 * p * d
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
            z[yi : yi + d] = obs[end + i - p][:, None]

            if i < p - 1:
                # we need one fewer controls than observations in the match region
                ui = p * d + i * c
                Z[ui : ui + c] = ctrl[i : i + n].T
                z[ui : ui + c] = ctrl[end + i - p][:, None]

            yi = matchdim + i * d
            Z[yi : yi + d] = obs[l + i : l + i + n].T

        for i in range(p - 1, p + l - 1):
            ui = matchdim + p * d + (i - p + 1) * c
            Z[ui : ui + c] = ctrl[i : i + n].T

        # average if needed
        if self.noise_handling == "average":
            n_columns = (l - p - 1) * c
            if n < n_columns:
                raise ValueError(f"history_length too short for noise averaging.")
            bins = torch.linspace(0, n, n_columns + 1, dtype=int)

            Z0 = Z
            Z = torch.empty((zdim, n_columns), dtype=dtype)
            for i, (i0, i1) in enumerate(zip(bins, bins[1:])):
                Z[:, i] = Z0[:, i0:i1].mean(dim=1)
        elif self.noise_handling != "none":
            raise ValueError(f"Unknown noise handling method: {self.noise_handling}.")

        # weigh using the appropriate coefficients
        Z_weighted = Z.clone()
        z_weighted = z.clone()

        if not self.exact_match:
            Z_weighted[: p * d] *= self.observation_match_cost
            z_weighted[: p * d] *= self.observation_match_cost

            Z_weighted[p * d : matchdim] *= self.control_match_cost
            z_weighted[p * d : matchdim] *= self.control_match_cost

        Z_weighted[matchdim : matchdim + p * d] *= self.target_cost
        Z_weighted[-l * c :] *= self.control_cost

        return Z, z, Z_weighted, z_weighted

    def _solve_gd(
        self, Z_weighted: torch.Tensor, z_weighted: torch.Tensor, U_unk: torch.Tensor
    ) -> torch.Tensor:
        # if self._previous_coeffs is None:
        #     coeffs = torch.zeros((self.history_length, 1))
        # else:
        #     coeffs = torch.vstack((self._previous_coeffs[1:], torch.tensor([[0.0]])))

        initial = torch.linalg.lstsq(Z_weighted, z_weighted)
        coeffs = initial.solution

        # TODO: smarter convergence criterion
        coeffs.requires_grad_()
        self._loss_curve = []
        optimizer = torch.optim.SGD([coeffs], lr=self.gd_lr)
        for i in range(self.gd_iterations):
            loss_quadratic = 0.5 * torch.sum((Z_weighted @ coeffs - z_weighted) ** 2)

            control_plan = U_unk @ coeffs
            loss_l1 = self.control_sparsity * control_plan.abs().sum()

            loss = loss_quadratic + loss_l1

            coeffs.grad = None
            loss.backward()

            optimizer.step()

            self._loss_curve.append(loss.item())

        coeffs = coeffs.detach()
        self._previous_coeffs = coeffs
        return coeffs

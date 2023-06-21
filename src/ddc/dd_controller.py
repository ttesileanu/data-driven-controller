import torch

from typing import Tuple, Optional
from types import SimpleNamespace

from .solvers import lstsq_constrained, lstsq_l1reg_constrained

# TODO: affine system?


class DDController:
    def __init__(
        self,
        observation_dim: int,
        control_dim: int,
        seed_length: int = 1,
        control_horizon: int = 1,
        state_dim: Optional[int] = None,
        history_length: Optional[int] = None,
        averaging_factor: float = 1.0,
        output_cost: float = 1.0,
        control_cost: float = 1.0,
        control_sparsity: float = 0.0,
        seed_slack_cost: float = 0.0,
        target_output: Optional[torch.Tensor] = None,
        method: str = "lstsq",
        gd_lr: float = 0.01,
        gd_iterations: int = 50,
        noise_handling: str = "none",
        offline: bool = False,
        noise_strength: float = 0.02,
        noise_policy: str = "online",
    ):
        """Data-driven controller.

        This is based on DeePC (Coulson et al., 2019), with some additions related to a
        similar method we developed before learning of DeePC.

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

        The representation shown above works in general provided that the data sequence
        is sufficiently diverse to sample the entire behavioral repertoire of the
        syste. When using a single long sequence to form the matrices `Y` and `U`,
        Willems' Lemma shows that the data-driven representations works as long as a
        property called `persistency of excitation` holds. In our case, this condition
        is related to a "tall" control Hankel matrix. Namely, if `U` has `l` rows, we
        need to focus on a matrix `Uext` with `n + l` rows, where `n` is the
        dimensionality of the (unobservable) system state. The data-driven
        representation works as long as `Uext` has full row rank, `c * (n + l)` (where
        `c` is the dimensionality of control).

        Our goal is to establish a control plan that maximally reduces the magnitude of
        the observations after a certain horizon, using the earlier measurements to
        predict the behavior of the system. To perform this prediction we need to start
        with a "seed" where the observations and control are known. The seed should be
        sufficiently long to fully specify the (unobserved) internal system state.

        More specifically, suppose the system's state is fully identified by observing
        `seed_length` steps and suppose that we're trying to minimize the system's
        output a number `control_horizon` of steps into the future. The number of prior
        samples used in the prediction is given by `history_length`.

        The control planning is set up as an optimization problem where the coefficients
        `alpha` are inferred such that the following loss function is minimized, subject
        to a constraint:

            L = output_cost * (predicted observations) ** 2
              + control_cost * (predicted control) ** 2
              + control_sparsity * L1_norm(predicted control) ,

        subject to

            predicted - measured seed observations = 0 , and
            predicted - measured seed controls = 0 .

        If observations are noisy, a certain amount of slack can be allowed in the seed
        observations. This removes the first set of contraints above, and instead adds a
        term

            seed_slack_cost * (measured seed observations) ** 2

        in the objective function.

        The inferred coefficients are then used to predict the optimal control. Note
        that apart from the L1 regularization term, this is just a least-squares
        problem with a linear constraint.

        By default the controller assumes that the dynamics is noiseless. If there is
        noise, the default controller will be suboptimal -- it will basically treat
        noise optimistically and assume that it will act in favor of stabilization. To
        treat noise more appropriately, set `noise_handling` to `"average"` or `"svd"`.
        The former averages over sets of columns of the Hankel matrices until the linear
        system that is solved has the number of equations plus the number of constraints
        equal to the number of unknowns. The `"svd"` option performs a similar
        reduction, but using SVD instead of averaging (following Zhang, Zheng, Li, 2022).

        Specifically, the number of columns is set to

            n_columns = trajectory_length * control_dim + state_dim * (control_dim + 1),

        where `trajectory_length = seed_length + control_horizon`.

        :param observation_dim: dimensionality of measurements
        :param control_dim: dimensionality of controls
        :param seed_length: how many measurements are expected to fix internal state;
            this is used to guess `state_dim` -- see below
        :param control_horizon: how many measurements and controls to predict
        :param state_dim: expected dimensionality of (unobserved) internal state; by
            default this is set to `observation_dim * seed_length`
        :param history_length: how many pairs of input/output samples to keep for
            determining dynamics; by default this is calculated as: (see also below)
                `int(self.minimal_history * averaging_factor)`
        :param averaging_factor: ratio between the size of the history and the minimal
            required number of samples; specifically, the history size is set to
                `int(self.minimal_history * averaging_factor)`
        :param output_cost: cost for difference between observations and target
        :param control_cost: cost for non-zero control (L2 norm)
        :param control_sparsity: L1 cost for control; only works when `method == "gd"`
        :param seed_slack_cost: cost for noise in observations during seed region
        :param target_output: target output value; default: zero
        :param method: can be
            "lstsq":    use `torch.linalg.lstsq`; does not support `control_sparsity`
            "gd":       gradient descent
        :param gd_lr: learning rate for `method == "gd"`
        :param gd_iterations: number of iterations when `method == "gd"`
        :param noise_handling: method for handling noisy data; can be
            "none":     assume noiseless observations
            "average":  average over samples; see above
            "svd":      reduce the number of columns in the Hankels using an SVD
                        (Zhang, Zheng, Li, 2022)
        :param offline: if true, the model freezes itself as soon as `history_length` is
            reached
        :param noise_strength: amount of noise to add to planned controls; this is
            relative to the magnitude of the last registered output
        :param noise_policy: when to add noise; can be:
            "always":   online, offline, and warm-up
            "online":   only during online and warm-up
        """
        self.observation_dim = observation_dim
        self.control_dim = control_dim
        self.seed_length = seed_length
        self.control_horizon = control_horizon

        if state_dim is None:
            self.state_dim = self.observation_dim * self.seed_length
        else:
            self.state_dim = state_dim

        self.averaging_factor = averaging_factor
        if history_length is None:
            self.history_length = int(self.minimal_history * averaging_factor)
        else:
            self.history_length = history_length

        self.output_cost = output_cost
        self.control_cost = control_cost
        self.control_sparsity = control_sparsity
        self.seed_slack_cost = seed_slack_cost
        self.target_output = target_output
        self.method = method
        self.gd_lr = gd_lr
        self.gd_iterations = gd_iterations
        self.noise_handling = noise_handling
        self.offline = offline
        self.noise_strength = noise_strength
        self.noise_policy = noise_policy

        if self.control_sparsity != 0:
            if self.method != "gd":
                raise ValueError("control_sparsity only works with `method=gd`")

        self.observation_history = None
        self.control_history = None

        self.frozen_observation_history = None
        self.frozen_control_history = None

        self.frozen = False

        self.history = SimpleNamespace(outputs=[], controls_prenoise=[], controls=[])

    def feed(self, observation: torch.Tensor):
        """Register another measurement.

        Up to `self.history_length` samples are kept.

        Control values preceding the observations are automatically stored based on the
        last output from `plan()`.

        :param observation: latest output from the model
        """
        assert len(observation) == self.observation_dim

        self.history.outputs.append(observation.clone())

        if len(self.history.controls) > 0:
            control = self.history.controls[-1]
        else:
            control = torch.zeros(self.control_dim, dtype=observation.dtype)

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

        self.observation_history = self.observation_history[-self.history_length :]
        self.control_history = self.control_history[-self.history_length :]

    def plan(self) -> torch.Tensor:
        """Estimate optimal control plan given the current history.

        :return: control plan, shape `(control_horizon, control_dim)`
        """
        h = self.control_horizon
        c = self.control_dim

        dtype = self.observation_history.dtype

        if len(self.observation_history) >= self.history_length:
            if self.offline:
                self.freeze()

            Zm, Zp, Zm_weighted, Zp_weighted = self.get_hankels()
            zm, zp, zm_weighted, zp_weighted = self.get_targets()

            U_unk = Zp[-h * c :]
            u_unk = zp[-h * c :]

            if self.method == "lstsq":
                coeffs = lstsq_constrained(
                    Zp_weighted, zp_weighted, Zm_weighted, zm_weighted
                )
            elif self.method == "gd":
                coeffs = lstsq_l1reg_constrained(
                    Zp_weighted,
                    zp_weighted,
                    Zm_weighted,
                    zm_weighted,
                    U_unk,
                    u_unk,
                    gamma=self.control_sparsity,
                    lr=self.gd_lr,
                    max_iterations=self.gd_iterations,
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # now solve z = Z alpha
            u_hat = U_unk @ coeffs

            control_plan_prenoise = u_hat.reshape((h, c))
        else:
            control_plan_prenoise = torch.zeros((h, c), dtype=dtype)

        self.history.controls_prenoise.append(control_plan_prenoise[0])

        have_noise = self.noise_strength > 0
        noise_policy = self.noise_policy == "always" or (
            self.noise_policy == "online" and not self.frozen
        )
        if have_noise and noise_policy:
            last_output = self.history.outputs[-1]
            eps = self.noise_strength * (
                2 * torch.rand(control_plan_prenoise.shape, dtype=dtype) - 1
            )
            noise = eps * torch.linalg.norm(last_output)
            control_plan = control_plan_prenoise + noise
        else:
            control_plan = control_plan_prenoise

        self.history.controls.append(control_plan[0])
        return control_plan

    def get_hankels(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate Hankel matrices.

        :return: tuple `(Zm, Zp, Zm_weighted, Zp_weighted)`, where `m` stands for match
            (in the seed region) and `p` stands for predict; weighted values are
            weighted by the (square roots of the) relevant costs
        """
        p = self.seed_length
        h = self.control_horizon
        l = p + h

        d = self.observation_dim
        c = self.control_dim

        obs, ctrl = self.get_past_trajectories()

        end = len(obs)
        assert len(ctrl) == end

        # populate Hankels
        n = end - l + 1
        dtype = obs.dtype
        Y = torch.empty((l * d, n), dtype=dtype)
        U = torch.empty((l * c, n), dtype=dtype)
        for i in range(l):
            yi = i * d
            Y[yi : yi + d] = obs[i : i + n].T

            ui = i * c
            U[ui : ui + c] = ctrl[i : i + n].T

        Uini = U[: p * c]
        Yini = Y[: p * d]
        Ypred = Y[p * d :]
        Upred = U[p * c :]
        Z = torch.vstack((Uini, Yini, Ypred, Upred))

        # reduce if needed
        if self.noise_handling != "none":
            n_columns = l * c + self.state_dim * (c + 1)
            if n < n_columns:
                raise ValueError(f"history too short for {self.noise_handling}.")

            Z0 = Z
            if self.noise_handling == "average":
                bins = torch.linspace(0, n, n_columns + 1, dtype=int)

                Z = torch.empty((len(Z), n_columns), dtype=dtype)
                for i, (i0, i1) in enumerate(zip(bins, bins[1:])):
                    Z[:, i] = Z0[:, i0:i1].mean(dim=1)
            elif self.noise_handling == "svd":
                if n < n_columns:
                    raise ValueError(f"history too short for SVD.")

                Zu, Zsv, _ = torch.linalg.svd(Z, full_matrices=True)
                Zs = torch.diag(Zsv)[:, :n_columns]
                Z = Zu @ Zs
            else:
                raise ValueError(
                    f"Unknown noise handling method: {self.noise_handling}."
                )

            # make sure the size of Z makes sense
            assert len(Z) == len(Z0)
            assert Z.shape[1] == n_columns

        # weigh using the appropriate coefficients
        Z_weighted = Z.clone()

        out_coeff = self.output_cost**0.5
        ctrl_coeff = self.control_cost**0.5
        matchdim = p * (c + d)
        Z_weighted[matchdim : matchdim + h * d] *= out_coeff
        Z_weighted[-h * c :] *= ctrl_coeff

        # handle seed slack cost, if any
        if self.seed_slack_cost > 0:
            Z_weighted[p * c : p * (c + d)] *= self.seed_slack_cost
            matchdim = p * c

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
        h = self.control_horizon
        l = p + h

        d = self.observation_dim
        c = self.control_dim

        obs = self.observation_history
        ctrl = self.control_history

        end = len(obs)
        assert len(ctrl) == end

        dtype = obs.dtype
        y = torch.zeros((l * d, 1), dtype=dtype)
        u = torch.zeros((l * c, 1), dtype=dtype)

        # initial values
        for i in range(p):
            yi = i * d
            y[yi : yi + d] = obs[end + i - p][:, None]

            ui = i * c
            u[ui : ui + c] = ctrl[end + i - p][:, None]

        # target values -- zero for control, but potentially non-zero for output
        if self.target_output is not None:
            for i in range(p, l):
                yi = i * d
                y[yi : yi + d] = self.target_output

        uini = u[: p * c]
        yini = y[: p * d]
        ypred = y[p * d :]
        upred = u[p * c :]
        z = torch.vstack((uini, yini, ypred, upred))

        # weigh using the appropriate coefficients
        z_weighted = z.clone()

        # handle seed slack cost, if any
        if self.seed_slack_cost > 0:
            z_weighted[p * c : p * (c + d)] *= self.seed_slack_cost
            matchdim = p * c
        else:
            matchdim = p * (c + d)

        zm = z[:matchdim]
        zp = z[matchdim:]
        zm_weighted = z_weighted[:matchdim]
        zp_weighted = z_weighted[matchdim:]
        return zm, zp, zm_weighted, zp_weighted

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
        """The minimal number of steps needed to achieve the required persistency of
        excitation necessary for control.

        This is a guess based on the given dimensionality of the internal state of the
        system, and assuming the system to be controllable.
        """
        p = self.seed_length
        h = self.control_horizon
        l = p + h
        d = self.observation_dim
        c = self.control_dim
        n = self.state_dim

        return (c + 1) * (n + l) - 1

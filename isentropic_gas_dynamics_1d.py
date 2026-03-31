"""
isentropic_gas_dynamics_1d.py
1-D isentropic gas dynamics problem definition, converted from MATLAB.

Two components: [rho, u]
Pressure law:   p = press_const * rho^gamma
"""

import numpy as np
from problem_def import ProblemDef


class IsentropicGasDynamics1D(ProblemDef):
    """
    1-D isentropic (barotropic) gas dynamics with Gaussian density perturbation.

    Equations:
        rho_t + (rho*u)_x = 0
        u_t   + u*u_x + (1/rho)*dp/drho * rho_x = 0
    where  p = press_const * rho^gamma.
    """

    def __init__(self):
        super().__init__(
            n_coarse       = 401,
            n_comp         = 2,
            ref_fac        = 4,
            max_no_levs    = 10,
            order          = 4,
            error_analyzer = 'func_var',
            err_thr        = 0.001,
            margin         = 3,
            nu             = 0.0001,
            x_left         = -10.0,
            x_right        = +10.0,
            time_int_type  = 'imex_111',
            cfl            = 0.025,
            n_refine       = 5,
            n_out          = 40,
            t_end          = 20.0,
        )

        self.left_bc  = np.zeros(self.n_comp)
        self.right_bc = np.zeros(self.n_comp)
        self.left_bc_type  = 'dirichlet'
        self.right_bc_type = 'dirichlet'

        self.gamma       = 1.4
        self.press_const = 1.0
        self.amp         = 1.0
        self.x_shift     = 0.0
        self.width       = 0.5

    # ------------------------------------------------------------------

    def initial_condition(self, x_values):
        """Background density 1 + Gaussian perturbation; zero velocity."""
        x   = np.asarray(x_values, dtype=float)
        n_c = len(x)

        rho   = 1.0 + self.amp * np.exp(-((x - self.x_shift) / self.width) ** 2)
        u_vel = np.zeros(n_c)

        ic = np.zeros((n_c, self.n_comp))
        ic[:, 0] = rho
        ic[:, 1] = u_vel
        return ic

    def eval_explicit_term(self, i_lev, i_seg, n_comp, t, x_coord, der, sol):
        """
        Isentropic Euler explicit terms:
            rho_t = -(u * rho_x + rho * u_x)
            u_t   = -(u * u_x + dpdrho * rho_x / rho)
        """
        D1 = der.sp_mat_d1[i_lev][i_seg]

        rho = sol[:, 0]
        u   = sol[:, 1]

        rho_x = D1.dot(rho)
        u_x   = D1.dot(u)

        dpdrho = self.press_const * self.gamma * rho ** (self.gamma - 1)

        td = np.zeros_like(sol)
        td[:, 0] = u * rho_x + rho * u_x
        td[:, 1] = u * u_x + dpdrho * rho_x / rho
        return td

    def set_bc(self, t):
        """Homogeneous Dirichlet BCs: rho=1, u=0."""
        left_bc  = np.array([1.0, 0.0])
        right_bc = np.array([1.0, 0.0])
        return left_bc, right_bc

    def compute_wave_speeds(self, amr_arr):
        """Eigenvalues: u ± sound speed at every coarse point."""
        n_c   = amr_arr.n_coarse
        rho   = amr_arr.f_coarse[:, 0]
        u     = amr_arr.f_coarse[:, 1]

        press = self.press_const * rho ** self.gamma
        ss    = np.sqrt(self.gamma * press / rho)

        wv_sp = np.zeros((n_c, 2))
        wv_sp[:, 0] = u - ss
        wv_sp[:, 1] = u + ss
        return wv_sp

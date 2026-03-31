"""
ideal_gas_dynamics_1d_primitive.py
1-D ideal gas dynamics in primitive variables, converted from MATLAB.

Primitive variables: q = [rho, u, p]
"""

import numpy as np
from problem_def import ProblemDef


class IdealGasDynamics1DPrimitive(ProblemDef):
    """
    Sod shock-tube problem in primitive variables (rho, u, p).

    The explicit term includes extra nonlinear viscous corrections to
    improve the Rankine-Hugoniot condition.
    """

    def __init__(self):
        super().__init__(
            n_coarse       = 201,
            n_comp         = 3,
            ref_fac        = 4,
            max_no_levs    = 10,
            order          = 4,
            error_analyzer = 'func_var',
            err_thr        = 0.0005,
            margin         = 3,
            nu             = 0.0001,
            x_left         = -1.0,
            x_right        = +1.0,
            time_int_type  = 'imex_111',
            cfl            = 0.1,
            n_refine       = 5,
            n_out          = 20,
            t_end          = 1.0,
        )

        self.left_bc  = np.zeros(self.n_comp)
        self.right_bc = np.zeros(self.n_comp)
        self.left_bc_type  = 'dirichlet'
        self.right_bc_type = 'dirichlet'

        self.gamma = 1.4
        self.x_frac = 0.5
        self.x_diaphragm = self.x_left + self.x_frac * (self.x_right - self.x_left)

        # Left state
        self.rho_l = 1.0
        self.u_l   = 0.75
        self.p_l   = 1.0

        # Right state
        self.rho_r = 0.125
        self.u_r   = 0.0
        self.p_r   = 0.1

        self.width = 0.0001

    # ------------------------------------------------------------------

    def initial_condition(self, x_values):
        """Smoothed Sod shock tube in primitive variables."""
        x   = np.asarray(x_values, dtype=float)
        n_c = len(x)

        one_to_zero = 0.5 - 0.5 * np.tanh((x - self.x_diaphragm) / self.width)
        zero_to_one = 0.5 + 0.5 * np.tanh((x - self.x_diaphragm) / self.width)

        rho   = self.rho_l * one_to_zero + self.rho_r * zero_to_one
        u_vel = self.u_l   * one_to_zero + self.u_r   * zero_to_one
        press = self.p_l   * one_to_zero + self.p_r   * zero_to_one

        ic = np.zeros((n_c, self.n_comp))
        ic[:, 0] = rho
        ic[:, 1] = u_vel
        ic[:, 2] = press
        return ic

    def eval_explicit_term(self, i_lev, i_seg, n_comp, t, x_coord, der, sol):
        """
        Primitive-variable Euler equations with nonlinear viscous corrections:
            rho_t = -(u * rho_x + rho * u_x)
            u_t   = -(u * u_x + p_x/rho) + nu * (2 * rho_x/rho * u_x)
            p_t   = -(u * p_x + gamma * p * u_x) + nu*(gamma-1)*rho*u_x^2
        """
        D1 = der.sp_mat_d1[i_lev][i_seg]

        rho   = sol[:, 0]
        u     = sol[:, 1]
        press = sol[:, 2]

        rho_x   = D1.dot(rho)
        u_x     = D1.dot(u)
        press_x = D1.dot(press)

        td = np.zeros_like(sol)
        td[:, 0] = u * rho_x + rho * u_x
        td[:, 1] = (u * u_x + press_x / rho
                    - self.nu * 2.0 * rho_x / rho * u_x)
        td[:, 2] = (u * press_x + self.gamma * press * u_x
                    - self.nu * (self.gamma - 1) * rho * u_x ** 2)
        return td

    def set_bc(self, t):
        """Time-independent Dirichlet BCs from left/right states."""
        left_bc  = np.array([self.rho_l, self.u_l, self.p_l])
        right_bc = np.array([self.rho_r, self.u_r, self.p_r])
        return left_bc, right_bc

    def compute_wave_speeds(self, amr_arr):
        """Eigenvalues: u-a, u, u+a at every coarse point."""
        n_c   = amr_arr.n_coarse
        rho   = amr_arr.f_coarse[:, 0]
        u     = amr_arr.f_coarse[:, 1]
        press = amr_arr.f_coarse[:, 2]

        ss = np.sqrt(self.gamma * press / rho)
        wv_sp = np.zeros((n_c, 3))
        wv_sp[:, 0] = u - ss
        wv_sp[:, 1] = u
        wv_sp[:, 2] = u + ss
        return wv_sp

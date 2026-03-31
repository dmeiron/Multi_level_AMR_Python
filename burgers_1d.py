"""
burgers_1d.py
1-D Burgers equation problem definition, converted from MATLAB.

Subclasses ProblemDef — all parameters set in __init__, no arguments required.
"""

import numpy as np
from problem_def import ProblemDef


class Burgers1D(ProblemDef):
    """
    Problem definition for the 1-D viscous Burgers equation:

        u_t + u * u_x = nu * u_xx

    with Gaussian initial condition and homogeneous Dirichlet BCs.
    """

    def __init__(self):
        super().__init__(
            n_coarse       = 101,
            n_comp         = 1,
            ref_fac        = 4,
            max_no_levs    = 10,
            order          = 4,
            error_analyzer = 'func_var',
            err_thr        = 0.001,
            margin         = 2,
            nu             = 0.001,
            x_left         = -1.0,
            x_right        = +9.0,
            time_int_type  = 'imex_111',
            cfl            = 0.1,
            n_refine       = 5,
            n_out          = 20,
            t_end          = 8.0,
        )

        self.left_bc  = np.zeros(self.n_comp)
        self.right_bc = np.zeros(self.n_comp)
        self.left_bc_type  = 'dirichlet'
        self.right_bc_type = 'dirichlet'

        # Problem-specific parameters
        self.x_shift = 0.5
        self.width   = 0.25

    # ------------------------------------------------------------------

    def initial_condition(self, x_values):
        """Gaussian bump centred at x_shift with half-width `width`."""
        x = np.asarray(x_values)
        ic = np.zeros((len(x), self.n_comp))
        ic[:, 0] = np.exp(-((x - self.x_shift) / self.width) ** 2)
        return ic

    def eval_explicit_term(self, i_lev, i_seg, n_comp, t, x_coord, der, sol):
        """Explicit (nonlinear advection) term:  rhs = u * u_x."""
        D1 = der.sp_mat_d1[i_lev][i_seg]
        u_x = D1.dot(sol)                   # shape (n_vars, n_comp)
        return sol * u_x

    def set_bc(self, t):
        """Homogeneous Dirichlet BCs (independent of time)."""
        left_bc  = np.zeros(self.n_comp)
        right_bc = np.zeros(self.n_comp)
        return left_bc, right_bc

    def compute_wave_speeds(self, amr_arr):
        """Wave speed = u (Burgers characteristic speed)."""
        return amr_arr.f_coarse[:, 0:1].copy()   # shape (n_coarse, 1)

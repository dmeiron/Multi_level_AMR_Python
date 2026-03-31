"""
ideal_gas_dynamics_1d_conservative.py
1-D ideal gas dynamics in conservative form (Euler equations), converted from MATLAB.

Conserved variables:  q = [mass, momentum, energy]
"""

import numpy as np
from problem_def import ProblemDef


class IdealGasDynamics1DConservative(ProblemDef):
    """
    Sod shock-tube problem in conservative variables.

    Conserved vector: [rho, rho*u, E]
    where E = p/(gamma-1) + 0.5*rho*u^2
    """

    def __init__(self):
        super().__init__(
            n_coarse       = 101,
            n_comp         = 3,
            ref_fac        = 4,
            max_no_levs    = 10,
            order          = 4,
            error_analyzer = 'lag_err_bound',
            err_thr        = 0.001,
            margin         = 3,
            nu             = 0.001,
            x_left         = -1.0,
            x_right        = +1.0,
            time_int_type  = 'imex_111',
            cfl            = 0.1,
            n_refine       = 3,
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
        self.rho_l = 2.0
        self.u_l   = 0.0
        self.p_l   = 10.0

        # Right state
        self.rho_r = 1.0
        self.u_r   = 0.0
        self.p_r   = 1.0

        self.width = 0.0005   # smoothing width for initial condition

    # ------------------------------------------------------------------
    # Required interface methods
    # ------------------------------------------------------------------

    def initial_condition(self, x_values):
        """Smoothed Sod shock tube in conservative variables."""
        x   = np.asarray(x_values, dtype=float)
        n_c = len(x)

        one_to_zero = 0.5 - 0.5 * np.tanh((x - self.x_diaphragm) / self.width)
        zero_to_one = 0.5 + 0.5 * np.tanh((x - self.x_diaphragm) / self.width)

        rho   = self.rho_l * one_to_zero + self.rho_r * zero_to_one
        u_vel = self.u_l   * one_to_zero + self.u_r   * zero_to_one
        press = self.p_l   * one_to_zero + self.p_r   * zero_to_one

        mass, momentum, energy = self.primitive_to_conserved(rho, u_vel, press)

        ic = np.zeros((n_c, self.n_comp))
        ic[:, 0] = mass
        ic[:, 1] = momentum
        ic[:, 2] = energy
        return ic

    def eval_explicit_term(self, i_lev, i_seg, n_comp, t, x_coord, der, sol):
        """Euler flux divergence (explicit nonlinear term)."""
        D1 = der.sp_mat_d1[i_lev][i_seg]

        mass     = sol[:, 0]
        momentum = sol[:, 1]
        energy   = sol[:, 2]

        rho, u_vel, press = self.conserved_to_primitive(mass, momentum, energy)

        f1 = rho * u_vel
        f2 = rho * u_vel ** 2 + press
        f3 = u_vel * energy + press * u_vel

        td = np.zeros_like(sol)
        td[:, 0] = D1.dot(f1)
        td[:, 1] = D1.dot(f2)
        td[:, 2] = D1.dot(f3)
        return td

    def set_bc(self, t):
        """Dirichlet BCs from left/right initial states (time-independent)."""
        m_l   = self.rho_l
        m_r   = self.rho_r
        mom_l = self.rho_l * self.u_l
        mom_r = self.rho_r * self.u_r
        e_l   = self.p_l / (self.gamma - 1) + 0.5 * self.rho_l * self.u_l ** 2
        e_r   = self.p_r / (self.gamma - 1) + 0.5 * self.rho_r * self.u_r ** 2

        left_bc  = np.array([m_l,  mom_l, e_l])
        right_bc = np.array([m_r,  mom_r, e_r])
        return left_bc, right_bc

    def compute_wave_speeds(self, amr_arr):
        """Eigenvalues: u-a, u, u+a at every coarse point."""
        n_c      = amr_arr.n_coarse
        mass     = amr_arr.f_coarse[:, 0]
        momentum = amr_arr.f_coarse[:, 1]
        energy   = amr_arr.f_coarse[:, 2]

        rho, u, press = self.conserved_to_primitive(mass, momentum, energy)
        ss = np.sqrt(self.gamma * press / rho)

        wv_sp = np.zeros((n_c, 3))
        wv_sp[:, 0] = u - ss
        wv_sp[:, 1] = u
        wv_sp[:, 2] = u + ss
        return wv_sp

    # ------------------------------------------------------------------
    # Primitive ↔ conserved conversions
    # ------------------------------------------------------------------

    def primitive_to_conserved(self, rho, u_vel, press):
        mass     = np.asarray(rho,   dtype=float)
        momentum = mass * np.asarray(u_vel,  dtype=float)
        energy   = (np.asarray(press, dtype=float) / (self.gamma - 1)
                    + 0.5 * mass * np.asarray(u_vel, dtype=float) ** 2)
        return mass, momentum, energy

    def conserved_to_primitive(self, mass, momentum, energy):
        mass     = np.asarray(mass,     dtype=float)
        momentum = np.asarray(momentum, dtype=float)
        energy   = np.asarray(energy,   dtype=float)
        rho    = mass
        u_vel  = momentum / mass
        press  = (self.gamma - 1) * (energy - 0.5 * mass * (momentum / mass) ** 2)
        return rho, u_vel, press

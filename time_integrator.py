"""
time_integrator.py
Time integration class, converted from MATLAB.

Implements IMEX-111 (forward Euler explicit / backward Euler implicit) on
each AMR level and segment.

Key mapping notes:
  - MATLAB lu() → scipy.sparse.linalg.splu()
  - MATLAB sparse identity + arithmetic → scipy.sparse operations
  - MATLAB cell arrays of sparse matrices → nested Python lists (same as fd_deriv.py)
  - All indexing 0-based
"""

import numpy as np
from scipy.sparse import eye as speye, csr_matrix
from scipy.sparse.linalg import splu

from fd_deriv import FDDeriv


class TimeIntegrator:
    """
    Container for time integration schemes on an AMR grid.

    Parameters
    ----------
    prob_def : object
        Must expose: .nu (viscosity), .order (FD order), .time_int_type (str)
    """

    def __init__(self, prob_def):
        self.nu         = prob_def.nu
        self.int_method = prob_def.time_int_type
        self.der        = FDDeriv(prob_def.order)

        # These are populated by generate_sparse_matrices()
        self.sp_mat_fd = None   # first-derivative sparse matrices [lev][seg]
        self.sp_mat_is = None   # implicit-solve sparse matrices   [lev][seg]

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

    def integrate(self, t: float, dt_c: float, u_at_t, prob_def):
        """
        Advance the solution one coarse time step dt_c.

        Parameters
        ----------
        t        : float    – current time
        dt_c     : float    – coarse time step
        u_at_t   : AMRArray – solution at time t
        prob_def : object   – problem definition

        Returns
        -------
        u_at_tpdt : AMRArray – solution at time t + dt_c
        """
        if self.int_method == 'imex_111':
            return self._imex_111(t, dt_c, u_at_t, prob_def)
        else:
            raise NotImplementedError(f"Method '{self.int_method}' is not yet supported.")

    def _imex_111(self, t: float, dt_c: float, u_at_t, prob_def):
        """
        IMEX-111: forward Euler (explicit nonlinear) + backward Euler (implicit viscous).
        Finer levels use proportionally smaller sub-steps (sub-cycling).
        """
        import copy
        assert self.sp_mat_is is not None, "Call generate_sparse_matrices() before integrate()"
        u_at_tpdt = copy.deepcopy(u_at_t)
        n_comp    = u_at_t.n_comp

        for i_lev in range(u_at_t.ref_levs_so_far + 1):

            n_seg   = 1 if i_lev == 0 else u_at_t.n_ref_seg[i_lev - 1]
            n_steps = u_at_t.ref_fac ** i_lev
            dt_f    = dt_c / n_steps

            for i_seg in range(n_seg):

                # Determine the index range and number of DOFs
                if i_lev == 0:
                    i_start = 0
                    i_end   = u_at_t.n_coarse - 2
                    n_vars  = u_at_t.n_coarse
                else:
                    i_start = u_at_t.beg_ref_seg[i_seg, i_lev - 1]
                    i_end   = u_at_t.end_ref_seg[i_seg, i_lev - 1]
                    n_vars  = (i_end - i_start + 1) * u_at_t.n_ref[i_lev - 1] + 1

                # Retrieve and factorise the implicit matrix for this segment
                sp_mat_lev_seg = self.sp_mat_is[i_lev][i_seg]
                assert sp_mat_lev_seg is not None, \
                    f"sp_mat_is[{i_lev}][{i_seg}] not built — call generate_sparse_matrices() first"
                lu_factor      = splu(sp_mat_lev_seg.tocsc())

                # Gather initial solution at this level/segment
                sol = np.zeros((n_vars, n_comp))
                for i_comp in range(n_comp):
                    sol[:, i_comp] = u_at_t.get_contiguous_refinement_array(
                        i_lev, i_comp, i_start, i_end
                    )

                x_coord = u_at_t.get_contiguous_refinement_x_coord(
                    i_lev, i_start, i_end
                )

                # Boundary conditions
                if i_lev == 0:
                    lbc_at_t,   rbc_at_t   = prob_def.set_bc(t)
                    lbc_at_tpdt, rbc_at_tpdt = prob_def.set_bc(t + dt_c)
                else:
                    n_p = u_at_t.n_ref[i_lev - 1]
                    lbc_at_t    = u_at_t.f_arr.get((i_start, i_lev - 1))[0, :]
                    lbc_at_tpdt = u_at_tpdt.f_arr.get((i_start, i_lev - 1))[0, :]
                    rbc_at_t    = u_at_t.f_arr.get((i_end, i_lev - 1))[n_p, :]
                    rbc_at_tpdt = u_at_tpdt.f_arr.get((i_end, i_lev - 1))[n_p, :]

                # Sub-step loop
                for i_step in range(1, n_steps + 1):
                    t_frac = i_step / n_steps

                    rhs = prob_def.eval_explicit_term(
                        i_lev, i_seg, n_comp, t, x_coord, self.der, sol
                    )
                    rhs = dt_f * rhs - sol   # shape (n_vars, n_comp)

                    for i_comp in range(n_comp):
                        left_bc  = ((1 - t_frac) * lbc_at_t[i_comp]
                                    + t_frac      * lbc_at_tpdt[i_comp])
                        right_bc = ((1 - t_frac) * rbc_at_t[i_comp]
                                    + t_frac      * rbc_at_tpdt[i_comp])

                        rhs[0,        i_comp] = left_bc
                        rhs[n_vars-1, i_comp] = right_bc

                        sol[:, i_comp] = lu_factor.solve(rhs[:, i_comp])

                # Store updated solution
                for i_comp in range(n_comp):
                    if i_lev == 0:
                        u_at_tpdt.set_coarse_array(sol[:, i_comp], i_comp)
                        u_at_tpdt.set_contiguous_refinement_array(
                            0, i_comp, 0, u_at_t.n_coarse - 2, sol[:, i_comp]
                        )
                    else:
                        u_at_tpdt.set_contiguous_refinement_array(
                            i_lev, i_comp, i_start, i_end, sol[:, i_comp]
                        )

        # Synchronise fine → coarse
        u_at_tpdt.fine_to_coarse()
        return u_at_tpdt

    # ------------------------------------------------------------------
    # Sparse matrix assembly
    # ------------------------------------------------------------------

    def generate_sparse_matrices(self, dt_c: float, amr_obj):
        """
        Build the implicit-solve sparse matrices for all levels and segments.

        For IMEX-111 the matrix at each level l, segment s is:
            D_2 = nu * dt_l * second_deriv_matrix - I
        with the first and last rows replaced by identity (Dirichlet BCs).

        Also stores the first-derivative matrices in self.sp_mat_fd.

        Parameters
        ----------
        dt_c    : float    – coarse time step
        amr_obj : AMRArray

        Mutates self.sp_mat_fd and self.sp_mat_is.
        Returns self for chaining.
        """
        # Build derivative matrices via FDDeriv
        self.der       = self.der.form_deriv_matrices(amr_obj)
        self.sp_mat_fd = self.der.sp_mat_d1
        self.sp_mat_is = self.der.sp_mat_d2   # will be overwritten below

        n_levels = amr_obj.ref_levs_so_far + 1

        # Re-initialise sp_mat_is as a fresh nested list
        self.sp_mat_is = [None] * n_levels
        self.sp_mat_is[0] = [None]
        for i_lev in range(1, n_levels):
            n_seg = amr_obj.n_ref_seg[i_lev - 1]
            self.sp_mat_is[i_lev] = [None] * n_seg

        for i_lev in range(n_levels):
            dt_l  = dt_c / amr_obj.n_ref[i_lev]
            n_seg = 1 if i_lev == 0 else amr_obj.n_ref_seg[i_lev - 1]

            for i_seg in range(n_seg):
                if i_lev == 0:
                    n_points = amr_obj.n_coarse
                else:
                    beg      = amr_obj.beg_ref_seg[i_seg, i_lev - 1]
                    end      = amr_obj.end_ref_seg[i_seg, i_lev - 1]
                    n_points = (end - beg + 1) * amr_obj.n_ref[i_lev] + 1

                assert self.der.sp_mat_d2 is not None, \
                    "der.sp_mat_d2 not built — form_deriv_matrices() must run first"
                sp_d2 = self.der.sp_mat_d2[i_lev][i_seg]
                assert sp_d2 is not None, f"sp_mat_d2[{i_lev}][{i_seg}] was not populated"
                I_mat = speye(n_points, format='csr')

                D2 = (self.nu * dt_l) * sp_d2 - I_mat
                D2 = D2.tolil()   # allow row assignment

                # Enforce Dirichlet BCs: first and last rows → identity
                D2[0, :]          = 0
                D2[n_points-1, :] = 0
                D2[0, 0]                    = 1.0
                D2[n_points-1, n_points-1]  = 1.0

                self.sp_mat_is[i_lev][i_seg] = csr_matrix(D2)

        return self
"""
error_measure.py
Error indicator class for AMR, converted from MATLAB.

Inherits from AMRArray and adds error estimation methods.

Indexing: all 0-based throughout.

Persistent MATLAB variables are handled as module-level caches (same pattern
as compute_derivative.py).
"""

import numpy as np
from math import factorial
from amr_array import AMRArray
import copy

# Module-level caches replacing MATLAB `persistent` variables
_lp_bound_cache: dict = {}       # keyed by order
_stencil_cache:  dict = {}       # keyed by order


class ErrorMeasure(AMRArray):
    """
    Error indicator derived from an AMRArray.

    Parameters
    ----------
    obj_in   : AMRArray  – source array whose structure is copied
    prob_def : object    – problem definition; must expose .order
    """

    def __init__(self, obj_in: AMRArray, prob_def):
        super().__init__(prob_def)

        # Copy refinement structure from source array
        self.ref_levs_so_far  = obj_in.ref_levs_so_far
        self.is_ref           = obj_in.is_ref.copy()
        self.n_ref_seg        = obj_in.n_ref_seg.copy()
        self.beg_ref_seg      = obj_in.beg_ref_seg.copy()
        self.end_ref_seg      = obj_in.end_ref_seg.copy()
        self.tot_ref_lev      = obj_in.tot_ref_lev.copy()
        self.coarse_lev_depth = obj_in.coarse_lev_depth.copy()
        self.x_coord          = copy.deepcopy(obj_in.x_coord)

        self.order   = prob_def.order
        # err_max[i, lev] stores the maximum error indicator at coarse point i, level lev
        self.err_max = np.zeros((self.n_coarse, self.max_no_levs))

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def analyze(self, obj_in: AMRArray, prob_def):
        """
        Compute the error indicator selected by prob_def.error_analyzer.

        Parameters
        ----------
        obj_in   : AMRArray  – the solution array to analyse
        prob_def : problem definition object

        Returns self (mutated in place).
        """
        method = prob_def.error_analyzer
        if method == 'func_var':
            return self.func_var(obj_in)
        elif method == 'lag_err_bound':
            return self.lag_err_bound(obj_in)
        else:
            raise ValueError(f"Error analyser '{method}' is not supported.")

    # ------------------------------------------------------------------
    # func_var: |df/dx| * dx as the error indicator
    # ------------------------------------------------------------------

    def func_var(self, obj_in: AMRArray):
        """
        Error indicator = |first derivative| * dx at each level.
        Uses compute_derivative_no_sync (imported lazily to avoid circular deps).
        """
        from compute_derivative_variants import compute_derivative_no_sync

        obj_der = compute_derivative_no_sync(obj_in, self.order, 1)

        for i_coarse in range(obj_in.n_coarse - 1):
            for i_lev in range(obj_in.coarse_lev_depth[i_coarse] + 1):
                arr = obj_der.f_arr.get((i_coarse, i_lev))
                if arr is None:
                    continue
                result = np.zeros_like(arr)
                for i_comp in range(obj_in.n_comp):
                    f = arr[:, i_comp] * self.dx[i_lev]
                    result[:, i_comp] = np.abs(f)
                self.f_arr[(i_coarse, i_lev)] = result
                self.err_max[i_coarse, i_lev] = np.max(
                    result[:, : obj_in.n_comp]
                )

        return self

    # ------------------------------------------------------------------
    # lag_err_bound: Lagrange interpolation error bound
    # ------------------------------------------------------------------

    def lag_err_bound(self, obj_in: AMRArray):
        """
        Error indicator based on the Lagrange interpolation error formula.
        Uses the (order+1)-th divided difference multiplied by a precomputed
        bound on the nodal polynomial.
        """
        global _lp_bound_cache

        if self.order not in _lp_bound_cache:
            _lp_bound_cache[self.order] = self._lag_bound()
        lp_bound = _lp_bound_cache[self.order]

        fact = factorial(self.order + 1)

        # --- Level 0 (coarse) ---
        i_start = 0
        i_end   = self.n_coarse - 2

        for i_comp in range(self.n_comp):
            values = obj_in.get_contiguous_refinement_array(0, i_comp, i_start, i_end)
            bound  = np.abs(self._compute_np1_divided_difference(values))
            bound  = bound / fact * lp_bound
            bound  = self._smooth(bound, i_end - i_start)
            self.set_contiguous_refinement_array(0, i_comp, i_start, i_end, bound)

        # --- Refined levels ---
        for i_lev in range(1, obj_in.ref_levs_so_far + 1):
            n_seg = obj_in.n_ref_seg[i_lev - 1]

            for i_seg in range(n_seg):
                i_start = obj_in.beg_ref_seg[i_seg, i_lev - 1]
                i_end   = obj_in.end_ref_seg[i_seg, i_lev - 1]

                for i_comp in range(self.n_comp):
                    values = obj_in.get_contiguous_refinement_array(
                        i_lev, i_comp, i_start, i_end
                    )
                    bound = np.abs(self._compute_np1_divided_difference(values))
                    bound = bound / fact * lp_bound
                    bound = self._smooth(bound, i_end - i_start)
                    self.set_contiguous_refinement_array(
                        i_lev, i_comp, i_start, i_end, bound
                    )

        # --- Compute per-point maxima ---
        for i_coarse in range(obj_in.n_coarse - 1):
            for i_lev in range(obj_in.coarse_lev_depth[i_coarse] + 1):
                arr = self.f_arr.get((i_coarse, i_lev))
                if arr is not None:
                    self.err_max[i_coarse, i_lev] = np.max(
                        arr[:, : obj_in.n_comp]
                    )

        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth(bound, length):
        """3-point averaging to reduce oscillation (MATLAB loop equivalent)."""
        b = bound.copy()
        for i in range(1, length - 1):   # i_ave = 2 .. i_end-i_start in 1-based
            b[i] = (bound[i - 1] + 2.0 * bound[i] + bound[i + 1]) / 4.0
        return b

    def _compute_np1_divided_difference(self, f):
        """
        Compute the (order+1)-th finite difference of f using a compact stencil.
        Replaces MATLAB persistent stencil_np1.
        """
        global _stencil_cache

        if self.order not in _stencil_cache:
            _stencil_cache[self.order] = self._nth_derivative_matrix(self.order + 1)
        stencil = _stencil_cache[self.order]

        nel  = len(f)
        np1  = len(stencil)
        n2   = np1 // 2
        delta = np.zeros(nel)

        for i in range(nel):
            min_ind = i - n2
            max_ind = min_ind + np1 - 1
            if min_ind < 0:
                min_ind = 0
            elif max_ind >= nel:
                min_ind = nel - np1
            for j in range(np1):
                delta[i] += stencil[j] * f[min_ind + j]

        return delta

    @staticmethod
    def _nth_derivative_matrix(n: int):
        """
        Return the first row of the n-th derivative finite-difference matrix
        on n+1 equally spaced nodes (the stencil weights).
        """
        m = n + 1
        x = np.arange(1, m + 1, dtype=float)   # nodes 1 .. m
        D = np.zeros((m, m))

        for i in range(m):
            r = x - x[i]
            # Vandermonde matrix: V[j, k] = r[k]^j
            V = np.vander(r, N=m, increasing=True).T   # shape (m, m)
            rhs = np.zeros(m)
            rhs[n] = float(factorial(n))
            w = np.linalg.solve(V, rhs)
            D[i, :] = w

        # All rows are the same for a uniform stencil; return first row
        return D[0, :]

    def _lag_bound(self):
        """
        Compute max |prod_{j=0}^{order} (x - j)| over a dense grid on [0, order].
        """
        n   = self.order
        x   = np.linspace(0.0, float(n), 2000)
        f   = np.ones(2000)
        for j in range(n + 1):
            f *= (x - j)
        return float(np.max(f))

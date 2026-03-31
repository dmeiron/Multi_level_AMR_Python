"""
fd_deriv.py
Finite-difference derivative class, converted from MATLAB.

Key mapping notes:
  - All arrays are 0-indexed.
  - MATLAB sparse() → scipy.sparse.csr_matrix built from (row, col, data) triplets.
  - MATLAB cell arrays of sparse matrices → nested Python lists.
  - 'persistent' variable in compute_derivative → module-level singleton (see
    compute_derivative.py).
"""

import numpy as np
from scipy.sparse import csr_matrix


class FDDeriv:
    """
    Finite-difference derivative operators of arbitrary even order.

    Parameters
    ----------
    order : int
        Order of accuracy (must be even, e.g. 2, 4, 6).

    Attributes
    ----------
    weights  : (op1, op1) ndarray  – barycentric-style stencil weights
    D_1      : (op1, op1) ndarray  – un-normalised first-derivative matrix
    D_2      : (op1, op1) ndarray  – un-normalised second-derivative matrix
    DN_1     : (op1, op1) ndarray  – normalised first-derivative matrix
    DN_2     : (op1, op1) ndarray  – normalised second-derivative matrix
    sp_mat_d1 : nested list of csr_matrix  – sparse first-derivative matrices per level/segment
    sp_mat_d2 : nested list of csr_matrix  – sparse second-derivative matrices per level/segment
    """

    def __init__(self, order: int):
        self.order = order
        op1 = order + 1

        x = np.linspace(0.0, 1.0, op1)

        # ------------------------------------------------------------------
        # Barycentric weights
        # ------------------------------------------------------------------
        self.weights = np.zeros((op1, op1))
        for i in range(op1):
            imin, imax = self._stencil(i, op1, order)
            for j in range(imin, imax + 1):
                self.weights[i, j] = 1.0
                for k in range(imin, imax + 1):
                    if k != j:
                        self.weights[i, j] /= (x[j] - x[k])

        # ------------------------------------------------------------------
        # First-derivative matrix D_1
        # ------------------------------------------------------------------
        self.D_1 = np.zeros((op1, op1))
        for l in range(op1):
            imin, imax = self._stencil(l, op1, order)
            for j in range(imin, imax + 1):
                if j != l:
                    self.D_1[l, j] = (
                        self.weights[l, j] / self.weights[l, l] / (x[l] - x[j])
                    )
            # Diagonal: negative row sum over the stencil
            self.D_1[l, l] = -np.sum(self.D_1[l, imin : imax + 1])

        # ------------------------------------------------------------------
        # Second-derivative matrix D_2
        # ------------------------------------------------------------------
        # s[l] = sum_{k != l} w[l,k] / (x[l] - x[k])
        s = np.zeros(op1)
        for l in range(op1):
            imin, imax = self._stencil(l, op1, order)
            for k in range(imin, imax + 1):
                if k != l:
                    s[l] += self.weights[l, k] / (x[l] - x[k])

        self.D_2 = np.zeros((op1, op1))
        for l in range(op1):
            imin, imax = self._stencil(l, op1, order)
            for j in range(imin, imax + 1):
                if j != l:
                    self.D_2[l, j] = (
                        -2.0 * self.weights[l, j] / (x[l] - x[j]) ** 2
                        - 2.0 * self.D_1[l, j] * s[l]
                    ) / self.weights[l, l]
            self.D_2[l, l] = -np.sum(self.D_2[l, imin : imax + 1])

        # Normalised matrices are set by normalize_derivatives()
        self.DN_1 = None
        self.DN_2 = None

        # Sparse matrices are set by form_deriv_matrices()
        self.sp_mat_d1 = None
        self.sp_mat_d2 = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stencil(i, n, order):
        """Return (index_min, index_max) for a centred stencil of width order+1."""
        imin = i - order // 2
        imax = imin + order
        if imin < 0:
            imin = 0
            imax = order
        if imax >= n:
            imax = n - 1
            imin = n - 1 - order
        return imin, imax

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def normalize_derivatives(self, dx: float):
        """
        Build DN_1 and DN_2 by scaling D_1 and D_2 for mesh spacing dx.
        Returns self so calls can be chained.
        """
        fac1 = 1.0 / (self.order * dx)
        fac2 = fac1 ** 2
        self.DN_1 = self.D_1 * fac1
        self.DN_2 = self.D_2 * fac2
        return self

    def compute_first_derivative(self, n_values: int, f_values) -> np.ndarray:
        """
        Apply the finite-difference first-derivative stencil to f_values.

        Parameters
        ----------
        n_values : int        – number of grid points
        f_values : array-like – function values, length n_values

        Returns
        -------
        df_1 : 1-D ndarray, length n_values
        """
        f_values = np.asarray(f_values, dtype=float)
        op1  = self.order + 1
        fo2  = self.order // 2
        df_1 = np.zeros(n_values)

        # Top rows (irregular stencil)
        for i in range(fo2):
            for j in range(op1):
                df_1[i] += self.DN_1[i, j] * f_values[j]

        # Regular interior rows (use the centred stencil row fo2)
        for j in range(op1):
            offset = j - fo2
            for i in range(fo2, n_values - fo2):
                df_1[i] += self.DN_1[fo2, j] * f_values[i + offset]

        # Bottom rows (irregular stencil)
        offset = n_values - op1
        for i in range(n_values - fo2, n_values):
            ip = i - (n_values - op1)       # local row in the bottom stencil block
            for j in range(op1):
                df_1[i] += self.DN_1[ip, j] * f_values[offset + j]

        return df_1

    def compute_second_derivative(self, n_values: int, f_values) -> np.ndarray:
        """
        Apply the finite-difference second-derivative stencil to f_values.

        Parameters
        ----------
        n_values : int        – number of grid points
        f_values : array-like – function values, length n_values

        Returns
        -------
        df_2 : 1-D ndarray, length n_values
        """
        f_values = np.asarray(f_values, dtype=float)
        op1  = self.order + 1
        fo2  = self.order // 2
        df_2 = np.zeros(n_values)

        # Top rows (irregular stencil)
        for i in range(fo2):
            for j in range(op1):
                df_2[i] += self.DN_2[i, j] * f_values[j]

        # Regular interior rows
        for i in range(fo2, n_values - fo2):
            for j in range(op1):
                df_2[i] += self.DN_2[fo2, j] * f_values[i + j - fo2]

        # Bottom rows (irregular stencil)
        offset = n_values - op1
        for i in range(n_values - fo2, n_values):
            ip = i - (n_values - op1)
            for j in range(op1):
                df_2[i] += self.DN_2[ip, j] * f_values[offset + j]

        return df_2

    def form_deriv_matrices(self, amr_obj):
        """
        Build sparse first- and second-derivative matrices for every level
        and segment of an AMRArray object.

        The results are stored in self.sp_mat_d1 and self.sp_mat_d2, which
        are nested lists:
            sp_mat_d1[i_lev][i_seg]  (both 0-indexed)

        Parameters
        ----------
        amr_obj : AMRArray instance
        """
        op1      = self.order + 1
        fo2      = self.order // 2
        n_levels = amr_obj.ref_levs_so_far + 1

        # Outer list indexed by level (0 = coarse)
        self.sp_mat_d1 = [None] * n_levels
        self.sp_mat_d2 = [None] * n_levels

        # Level 0: single segment (the whole coarse grid)
        self.sp_mat_d1[0] = [None]
        self.sp_mat_d2[0] = [None]

        for i_lev in range(1, n_levels):
            n_seg = amr_obj.n_ref_seg[i_lev - 1]
            self.sp_mat_d1[i_lev] = [None] * n_seg
            self.sp_mat_d2[i_lev] = [None] * n_seg

        for i_lev in range(n_levels):
            self.normalize_derivatives(amr_obj.dx[i_lev])

            if i_lev == 0:
                segments = [(0, amr_obj.n_coarse)]   # (seg_idx, n_points)
            else:
                n_seg = amr_obj.n_ref_seg[i_lev - 1]
                segments = []
                for i_seg in range(n_seg):
                    beg = amr_obj.beg_ref_seg[i_seg, i_lev - 1]
                    end = amr_obj.end_ref_seg[i_seg, i_lev - 1]
                    n_pts = (end - beg + 1) * amr_obj.n_ref[i_lev] + 1
                    segments.append((i_seg, n_pts))

            for i_seg, n_points in segments:
                rows = []
                cols = []
                s1   = []
                s2   = []

                # Top rows (irregular stencil)
                for i in range(fo2):
                    for j in range(op1):
                        rows.append(i)
                        cols.append(j)
                        s1.append(self.DN_1[i, j])
                        s2.append(self.DN_2[i, j])

                # Regular interior rows
                for i in range(fo2, n_points - fo2):
                    for j in range(op1):
                        rows.append(i)
                        cols.append(i + j - fo2)
                        s1.append(self.DN_1[fo2, j])
                        s2.append(self.DN_2[fo2, j])

                # Bottom rows (irregular stencil)
                offset = n_points - op1
                for i in range(n_points - fo2, n_points):
                    ip = i - (n_points - op1)
                    for j in range(op1):
                        rows.append(i)
                        cols.append(offset + j)
                        s1.append(self.DN_1[ip, j])
                        s2.append(self.DN_2[ip, j])

                shape = (n_points, n_points)
                self.sp_mat_d1[i_lev][i_seg] = csr_matrix(
                    (s1, (rows, cols)), shape=shape
                )
                self.sp_mat_d2[i_lev][i_seg] = csr_matrix(
                    (s2, (rows, cols)), shape=shape
                )

        return self

"""
lagrange_interp.py
Lagrange interpolation class, converted from MATLAB.

Indexing notes:
  - All arrays are 0-indexed.
  - weights[i, j] corresponds to MATLAB's weights(i+1, j+1).
  - i_interp passed in by callers must be 0-indexed.
"""

import numpy as np


class LagrangeInterp:
    """
    Barycentric-style Lagrange interpolation on a uniform stencil.

    Parameters
    ----------
    order : int
        Polynomial order of the interpolant.
    """

    def __init__(self, order: int):
        self.order = order
        op1 = order + 1

        # Reference nodes on [0, 1]
        x = np.linspace(0.0, 1.0, op1)

        # weights[i, j] = product_{k != j} 1/(x[j] - x[k])
        # for the stencil centred on node i.
        self.weights = np.zeros((op1, op1))

        for i in range(op1):
            index_min, index_max = self._stencil(i, op1, order)

            for j in range(index_min, index_max + 1):
                self.weights[i, j] = 1.0
                for k in range(index_min, index_max + 1):
                    if k != j:
                        self.weights[i, j] /= (x[j] - x[k])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stencil(i, n, order):
        """Return (index_min, index_max) for a centred stencil of width order+1."""
        index_min = i - order // 2
        index_max = index_min + order
        if index_min < 0:
            index_min = 0
            index_max = order
        if index_max >= n:
            index_max = n - 1
            index_min = n - 1 - order
        return index_min, index_max

    def _ip_and_stencil(self, i_interp, num_points):
        """
        Compute the stencil bounds and local pointer ip for a given
        interpolation site i_interp (0-indexed).

        Returns (index_min, index_max, ip) all 0-indexed.
        """
        order = self.order
        index_min = i_interp - order // 2
        index_max = index_min + order

        if index_min < 0:
            index_min = 0
            index_max = order
            ip = i_interp - index_min          # local position of i_interp in stencil

        elif index_max >= num_points:
            index_max = num_points - 1
            index_min = num_points - 1 - order
            ip = i_interp - index_min

        else:
            ip = order // 2

        return index_min, index_max, ip

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def interpolate_at_reg_spaced_pts(self, x, f, i_interp, n_interp):
        """
        Interpolate f onto n_interp+1 equally spaced points inside
        the interval [x[i_interp], x[i_interp+1]].

        Parameters
        ----------
        x        : 1-D array of uniformly spaced nodes (0-indexed)
        f        : 1-D array of function values at x
        i_interp : int, 0-indexed left endpoint of the target interval
        n_interp : int, number of sub-intervals (output has n_interp+1 points)

        Returns
        -------
        x_interp : 1-D ndarray, length n_interp+1
        f_interp : 1-D ndarray, length n_interp+1
        """
        num_points = len(x)
        dx = x[1] - x[0]
        dx_interp = dx / n_interp

        x_interp = np.zeros(n_interp + 1)
        f_interp = np.zeros(n_interp + 1)

        index_min, index_max, ip = self._ip_and_stencil(i_interp, num_points)

        # First point is the node itself
        x_interp[0] = x[i_interp]
        f_interp[0] = f[i_interp]

        # Interior interpolated points
        for k in range(1, n_interp):
            x_p = x[i_interp] + k * dx_interp
            numer = 0.0
            denom = 0.0
            for j in range(index_min, index_max + 1):
                jp = j - index_min          # local index into stencil
                w  = self.weights[ip, jp]
                d  = x_p - x[j]
                numer += w * f[j] / d
                denom += w / d
            x_interp[k] = x_p
            f_interp[k] = numer / denom

        # Last point is the right endpoint of the interval
        x_interp[n_interp] = x[i_interp + 1]
        f_interp[n_interp] = f[i_interp + 1]

        return x_interp, f_interp

    def interp_at_selected_points(self, x, f, i_interp, x_interp):
        """
        Evaluate the Lagrange interpolant at arbitrary points x_interp,
        using a stencil centred near i_interp.

        Parameters
        ----------
        x        : 1-D array of uniformly spaced nodes (0-indexed)
        f        : 1-D array of function values at x
        i_interp : int, 0-indexed reference node for stencil selection
        x_interp : 1-D array of target x-values

        Returns
        -------
        f_interp : 1-D ndarray of interpolated values
        """
        num_points = len(x)
        n_interp   = len(x_interp)
        f_interp   = np.zeros(n_interp)

        index_min, index_max, ip = self._ip_and_stencil(i_interp, num_points)

        for k in range(n_interp):
            x_p   = x_interp[k]
            numer = 0.0
            denom = 0.0

            for j in range(index_min, index_max + 1):
                if abs(x_p - x[j]) > 10 * np.finfo(float).eps:
                    jp = j - index_min
                    w  = self.weights[ip, jp]
                    d  = x_p - x[j]
                    numer += w * f[j] / d
                    denom += w / d
                else:
                    # Evaluation exactly at a node — return nodal value
                    numer = f[j]
                    denom = 1.0
                    break

            f_interp[k] = numer / denom

        return f_interp

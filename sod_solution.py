"""
sod_solution.py
Sod shock-tube solution stub, converted from MATLAB.

The original MATLAB was an unimplemented stub (returned inputArg1/inputArg2
which were never defined).  This Python version is a proper stub that raises
NotImplementedError, signalling that the function needs to be implemented.

When you are ready to implement it you can either:
  - Fill in the body below, or
  - Delegate to euler_riemann_problem.euler_riemann_problem() which already
    provides a complete exact Riemann solver.
"""

import numpy as np


def sod_solution(rho_L, u_L, p_L,
                 rho_R, u_R, p_R,
                 t,
                 x_left, x_right, x_diaphragm,
                 n_points):
    """
    Compute the Sod shock-tube exact solution on a uniform grid.

    Parameters
    ----------
    rho_L, u_L, p_L : float – left state
    rho_R, u_R, p_R : float – right state
    t               : float – evaluation time
    x_left          : float – left boundary
    x_right         : float – right boundary
    x_diaphragm     : float – initial discontinuity location
    n_points        : int   – number of output points

    Returns
    -------
    rho, u_vel, p : 1-D ndarrays of length n_points

    Note
    ----
    This was a stub in the original MATLAB code. The implementation below
    delegates to euler_riemann_problem for a complete exact solution.
    """
    from euler_riemann_problem import euler_riemann_problem

    x_array = np.linspace(x_left, x_right, n_points)

    rho, u_vel, p = euler_riemann_problem(
        rho_L, u_L, p_L,
        rho_R, u_R, p_R,
        t,
        x_left, x_right, x_diaphragm,
        x_array, n_points
    )

    return rho, u_vel, p

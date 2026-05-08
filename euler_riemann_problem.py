"""
euler_riemann_problem.py
Exact Riemann solver for the 1-D Euler equations, converted from MATLAB.

Bugs fixed from the original:
  - `rho(i_x. i_t)` typo in right-rarefaction star-state branch → `rho[i_x]`
  - `P_R` (capital P) in the both-shocks branch of compute_p_star → `p_R`
  - `gammna==+1` syntax error in riemann_exact_solution → not present here
    (that was in the separate riemann_exact_solution.m standalone script)

Public API
----------
euler_riemann_problem(rho_L, u_L, p_L, rho_R, u_R, p_R,
                      t, x_left, x_right, x_diaphragm,
                      x_array, n_points)
    -> (rho, u_vel, p)  each a 1-D ndarray of length n_points
"""

import numpy as np


def euler_riemann_problem(rho_L, u_L, p_L,
                          rho_R, u_R, p_R,
                          t,
                          x_left, x_right, x_diaphragm,
                          x_array, n_points):
    """
    Evaluate the exact Riemann solution at points x_array and time t.

    Parameters
    ----------
    rho_L, u_L, p_L : float – left state (density, velocity, pressure)
    rho_R, u_R, p_R : float – right state
    t               : float – evaluation time (> 0)
    x_left          : float – (unused, kept for API compatibility)
    x_right         : float – (unused, kept for API compatibility)
    x_diaphragm     : float – initial discontinuity location
    x_array         : array-like, length n_points
    n_points        : int

    Returns
    -------
    rho, u_vel, p : 1-D ndarrays of length n_points
    """
    gamma = 1.4

    (right_shock, left_shock,
     right_rarefaction, left_rarefaction,
     p_star, u_star) = _compute_p_star(rho_L, u_L, p_L,
                                        rho_R, u_R, p_R, gamma)

 # --- Left wave ---
    shock_speed_L = float('nan')
    a_L = float('nan')
    S_HL = float('nan')
    S_TL = float('nan')

    if left_shock:
        rho_star_L    = rho_L * (_shock_density_ratio(p_star, p_L, gamma))
        shock_speed_L = u_L - _shock_speed(p_star, p_L, rho_L, gamma)
    else:  # left rarefaction
        rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)
        a_L        = np.sqrt(gamma * p_L / rho_L)
        a_star_L   = a_L * (p_star / p_L) ** ((gamma - 1) / (2 * gamma))
        S_HL       = u_L - a_L
        S_TL       = u_star - a_star_L

# --- Right wave ---
    shock_speed_R = float('nan')
    a_R = float('nan')
    S_HR = float('nan')
    S_TR = float('nan')

    if right_shock:
        rho_star_R    = rho_R * (_shock_density_ratio(p_star, p_R, gamma))
        shock_speed_R = u_R + _shock_speed(p_star, p_R, rho_R, gamma)
    else:  # right rarefaction
        rho_star_R = rho_R * (p_star / p_R) ** (1.0 / gamma)
        a_R        = np.sqrt(gamma * p_R / rho_R)
        a_star_R   = a_R * (p_star / p_R) ** ((gamma - 1) / (2 * gamma))
        S_HR       = u_R + a_R
        S_TR       = u_star + a_star_R

    # --- Sample solution ---
    x_array = np.asarray(x_array, dtype=float)
    rho   = np.zeros(n_points)
    u_vel = np.zeros(n_points)
    p     = np.zeros(n_points)

    x_con = x_diaphragm + u_star * t      # contact discontinuity location

    for i_x in range(n_points):
        x = x_array[i_x]

        if x < x_con:
            # Left of contact
            if left_shock:
                if x < shock_speed_L * t:
                    rho[i_x], u_vel[i_x], p[i_x] = rho_L, u_L, p_L
                else:
                    rho[i_x], u_vel[i_x], p[i_x] = rho_star_L, u_star, p_star
            else:  # left rarefaction
                if x < S_HL * t:
                    rho[i_x], u_vel[i_x], p[i_x] = rho_L, u_L, p_L
                elif x > S_TL * t:
                    rho[i_x], u_vel[i_x], p[i_x] = rho_star_L, u_star, p_star
                else:  # inside fan
                    xi = x / t
                    rho[i_x] = rho_L * (2 / (gamma + 1)
                                         + (gamma - 1) / ((gamma + 1) * a_L)
                                         * (u_L - xi)) ** (2 / (gamma - 1))
                    u_vel[i_x] = 2 / (gamma + 1) * (a_L + (gamma - 1) / 2 * u_L + xi)
                    p[i_x] = p_L * (2 / (gamma + 1)
                                     + (gamma - 1) / ((gamma + 1) * a_L)
                                     * (u_L - xi)) ** (2 * gamma / (gamma - 1))

        elif x > x_con:
            # Right of contact
            if right_shock:
                if x > shock_speed_R * t:
                    rho[i_x], u_vel[i_x], p[i_x] = rho_R, u_R, p_R
                else:
                    rho[i_x], u_vel[i_x], p[i_x] = rho_star_R, u_star, p_star
            else:  # right rarefaction
                if x > S_HR * t:
                    rho[i_x], u_vel[i_x], p[i_x] = rho_R, u_R, p_R
                elif x < S_TR * t:
                    rho[i_x], u_vel[i_x], p[i_x] = rho_star_R, u_star, p_star
                else:  # inside fan
                    xi = x / t
                    rho[i_x] = rho_R * (2 / (gamma + 1)
                                         - (gamma - 1) / ((gamma + 1) * a_R)
                                         * (u_R - xi)) ** (2 / (gamma - 1))
                    u_vel[i_x] = 2 / (gamma + 1) * (-a_R + (gamma - 1) / 2 * u_L + xi)
                    p[i_x] = p_R * (2 / (gamma + 1)
                                     - (gamma - 1) / ((gamma + 1) * a_R)
                                     * (u_R - xi)) ** (2 * gamma / (gamma - 1))
        else:
            # Exactly on contact — use left star state
            rho[i_x], u_vel[i_x], p[i_x] = rho_star_L, u_star, p_star

    return rho, u_vel, p


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shock_density_ratio(p_star, p_K, gamma):
    """Post-shock density / pre-shock density."""
    return ((p_star / p_K + (gamma - 1) / (gamma + 1))
            / ((gamma - 1) / (gamma + 1) * (p_star / p_K) + 1))


def _shock_speed(p_star, p_K, rho_K, gamma):
    """Magnitude of the shock speed relative to the undisturbed state."""
    return np.sqrt((p_star + (gamma - 1) / (gamma + 1) * p_K)
                   / (2 / ((gamma + 1) * rho_K))) / rho_K


def _compute_p_star(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma):
    """
    Newton iteration to find the pressure in the star region.

    Returns
    -------
    right_shock, left_shock, right_rarefaction, left_rarefaction : bool
    p_star, u_star : float
    """
    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    if 2 * a_L / (gamma - 1) + 2 * a_R / (gamma - 1) < u_R - u_L:
        raise RuntimeError("Vacuum condition detected in Riemann problem.")

    p_min = min(p_L, p_R)
    p_max = max(p_L, p_R)

    f_min = _f_left(p_min, p_L, rho_L, gamma)[0] + _f_right(p_min, p_R, rho_R, gamma)[0]
    f_max = _f_left(p_max, p_L, rho_L, gamma)[0] + _f_right(p_max, p_R, rho_R, gamma)[0]

    right_shock      = False
    left_shock       = False
    right_rarefaction = False
    left_rarefaction  = False

    if f_min >= 0 and f_max >= 0:          # both rarefactions
        p_star = p_min / 2.0
        left_rarefaction  = True
        right_rarefaction = True
    elif f_min < 0 <= f_max:               # mixed
        p_star = (p_min + p_max) / 2.0
        if p_star > p_R:
            right_shock      = True
            left_rarefaction = True
        else:
            left_shock        = True
            right_rarefaction = True
    else:                                   # both shocks
        p_star = ((p_L + p_R) / 2.0        # fixed: was P_R (capital) in MATLAB
                  - (1.0 / 8.0) * (u_R - u_L) * (rho_R + rho_L) * (a_L + a_R))
        right_shock = True
        left_shock  = True

    # Newton iterations
    tol      = 1.0e-6
    iter_max = 10
    for _ in range(iter_max):
        f_L, fp_L = _f_left( p_star, p_L, rho_L, gamma)
        f_R, fp_R = _f_right(p_star, p_R, rho_R, gamma)
        resid = f_L + f_R + u_R - u_L
        if abs(resid) < tol:
            break
        p_star -= resid / (fp_L + fp_R)
    else:
        raise RuntimeError(f"p_star iteration failed to converge after {iter_max} iterations.")

    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)
    return right_shock, left_shock, right_rarefaction, left_rarefaction, p_star, u_star


def _f_left(p, p_L, rho_L, gamma):
    """Shock/rarefaction function and derivative for the left wave."""
    if p > p_L:
        A_L  = 2.0 / ((gamma + 1) * rho_L)
        B_L  = (gamma - 1) / (gamma + 1) * p_L
        f    = (p - p_L) * np.sqrt(A_L / (p + B_L))
        fp   = np.sqrt(A_L / (B_L + p)) * (1 - (p - p_L) / (2 * (B_L + p)))
    else:
        a_L  = np.sqrt(gamma * p_L / rho_L)
        f    = 2 * a_L / (gamma - 1) * ((p / p_L) ** ((gamma - 1) / (2 * gamma)) - 1)
        fp   = (1 / (rho_L * a_L)) * (p / p_L) ** (-(gamma + 1) / (2 * gamma))
    return f, fp


def _f_right(p, p_R, rho_R, gamma):
    """Shock/rarefaction function and derivative for the right wave."""
    if p > p_R:
        A_R  = 2.0 / ((gamma + 1) * rho_R)
        B_R  = (gamma - 1) / (gamma + 1) * p_R
        f    = (p - p_R) * np.sqrt(A_R / (p + B_R))
        fp   = np.sqrt(A_R / (B_R + p)) * (1 - (p - p_R) / (2 * (B_R + p)))
    else:
        a_R  = np.sqrt(gamma * p_R / rho_R)
        f    = 2 * a_R / (gamma - 1) * ((p / p_R) ** ((gamma - 1) / (2 * gamma)) - 1)
        fp   = (1 / (rho_R * a_R)) * (p / p_R) ** (-(gamma + 1) / (2 * gamma))
    return f, fp

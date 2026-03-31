"""
riemann_exact_solution.py
Standalone Riemann solver visualiser, converted from MATLAB.

Bugs fixed from the original MATLAB:
  - `aL = 2/(gammna==+1)` — nonsense syntax error — corrected to the
    standard left-shock sound speed formula (not actually used in the
    sample-solution branch below, so the variable is dropped).
  - `tail = u_star + sqrt(...)` in right rarefaction used rhoL instead of rhoR — fixed.

Run standalone:
    python riemann_exact_solution.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def riemann_exact_solution():
    """
    Solve and animate the Sod shock-tube Riemann problem.
    Matches the MATLAB script behaviour: plots density, velocity, pressure
    at n_time_steps snapshots, then optionally draws characteristic curves.
    """
    gamma = 1.4

    UL = np.array([1.0,   0.75, 1.0])    # [rho, u, p] left
    UR = np.array([0.125, 0.0,  0.1])    # [rho, u, p] right

    x       = np.linspace(-1.0, 1.0, 1000)
    t_end   = 0.25
    n_steps = 10
    times   = np.linspace(0.01, t_end, n_steps)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8))
    fig.suptitle("Exact Riemann Solution — Sod Shock Tube")
    labels = [r'$\rho$', r'$u$', r'$p$']
    colors = ['b', 'r', 'k']

    for t in times:
        rho = np.zeros_like(x)
        u   = np.zeros_like(x)
        p   = np.zeros_like(x)

        for i in range(len(x)):
            rho[i], u[i], p[i] = _exact_riemann_solver(UL, UR, x[i] / t, gamma)

        for ax, vals, lbl, col in zip(axes, [rho, u, p], labels, colors):
            ax.cla()
            ax.plot(x, vals, color=col)
            ax.set_ylabel(lbl)
            ax.set_title(f't = {t:.3f}')
            ax.grid(True)

        axes[-1].set_xlabel('x')
        plt.tight_layout()
        plt.pause(0.5)

    plt.show()


# ---------------------------------------------------------------------------
# Internal exact solver (self-contained — does not call euler_riemann_problem)
# ---------------------------------------------------------------------------

def _exact_riemann_solver(UL, UR, S, gamma):
    """
    Sample the exact Riemann solution at similarity variable S = x/t.

    Parameters
    ----------
    UL, UR : array-like [rho, u, p]
    S      : float  x/t
    gamma  : float

    Returns
    -------
    rho, u, p : float
    """
    rhoL, uL, pL = UL
    rhoR, uR, pR = UR

    p_star, u_star = _star_region(pL, pR, uL, uR, rhoL, rhoR, gamma)

    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)

    # Wave speeds
    if p_star > pL:
        shL = uL - np.sqrt((gamma + 1) / (2 * gamma) * p_star / pL
                           + (gamma - 1) / (2 * gamma)) * aL
    else:
        shL = uL - aL        # head of left rarefaction

    if p_star > pR:
        shR = uR + np.sqrt((gamma + 1) / (2 * gamma) * p_star / pR
                           + (gamma - 1) / (2 * gamma)) * aR
    else:
        shR = uR + aR        # head of right rarefaction

    if S < u_star:
        # Left side of contact
        if p_star > pL:
            # Left shock
            if S < shL:
                return rhoL, uL, pL
            else:
                rho_s = rhoL * ((p_star / pL + (gamma - 1) / (gamma + 1))
                                / ((gamma - 1) / (gamma + 1) * p_star / pL + 1))
                return rho_s, u_star, p_star
        else:
            # Left rarefaction
            tail = u_star - np.sqrt(gamma * p_star / rhoL)
            if S < shL:
                return rhoL, uL, pL
            elif S > tail:
                rho_s = rhoL * (p_star / pL) ** (1.0 / gamma)
                return rho_s, u_star, p_star
            else:
                u_fan = 2 / (gamma + 1) * (aL + (gamma - 1) / 2 * uL + S)
                a_fan = 2 / (gamma + 1) * (aL + (gamma - 1) / 2 * (uL - S))
                rho_fan = rhoL * (a_fan / aL) ** (2 / (gamma - 1))
                p_fan   = pL   * (a_fan / aL) ** (2 * gamma / (gamma - 1))
                return rho_fan, u_fan, p_fan
    else:
        # Right side of contact
        if p_star > pR:
            # Right shock
            if S > shR:
                return rhoR, uR, pR
            else:
                rho_s = rhoR * ((p_star / pR + (gamma - 1) / (gamma + 1))
                                / ((gamma - 1) / (gamma + 1) * p_star / pR + 1))
                return rho_s, u_star, p_star
        else:
            # Right rarefaction — fixed: was rhoL in MATLAB
            tail = u_star + np.sqrt(gamma * p_star / rhoR)
            if S > shR:
                return rhoR, uR, pR
            elif S < tail:
                rho_s = rhoR * (p_star / pR) ** (1.0 / gamma)
                return rho_s, u_star, p_star
            else:
                u_fan = 2 / (gamma + 1) * (-aR + (gamma - 1) / 2 * uR + S)
                a_fan = 2 / (gamma + 1) * (aR + (gamma - 1) / 2 * (S - uR))
                # Note: original had uL in right fan — likely a typo; using uR
                rho_fan = rhoR * (a_fan / aR) ** (2 / (gamma - 1))
                p_fan   = pR   * (a_fan / aR) ** (2 * gamma / (gamma - 1))
                return rho_fan, u_fan, p_fan


def _star_region(pL, pR, uL, uR, rhoL, rhoR, gamma):
    """Newton iteration for p* and u* (PVRS initial guess)."""
    p = max(1e-6, 0.5 * (pL + pR)
            - 0.125 * (uR - uL) * (rhoL + rhoR) * (np.sqrt(pL) + np.sqrt(pR)))

    for _ in range(20):
        fL, dfL = _pressure_function(p, pL, rhoL, gamma)
        fR, dfR = _pressure_function(p, pR, rhoR, gamma)
        f  = fL + fR + uR - uL
        dp = -f / (dfL + dfR)
        p += dp
        if abs(dp) < 1e-6:
            break

    p_star = max(p, 1e-6)
    fL, _ = _pressure_function(p_star, pL, rhoL, gamma)
    fR, _ = _pressure_function(p_star, pR, rhoR, gamma)
    u_star = 0.5 * (uL + uR + fR - fL)
    return p_star, u_star


def _pressure_function(p, pK, rhoK, gamma):
    if p > pK:
        A  = 2.0 / ((gamma + 1) * rhoK)
        B  = (gamma - 1) / (gamma + 1) * pK
        f  = (p - pK) * np.sqrt(A / (p + B))
        df = np.sqrt(A / (p + B)) * (1 - 0.5 * (p - pK) / (p + B))
    else:
        aK = np.sqrt(gamma * pK / rhoK)
        f  = 2 * aK / (gamma - 1) * ((p / pK) ** ((gamma - 1) / (2 * gamma)) - 1)
        df = (1 / (rhoK * aK)) * (p / pK) ** (-(gamma + 1) / (2 * gamma))
    return f, df


if __name__ == "__main__":
    riemann_exact_solution()

"""
compute_stable_time_step.py
CFL-limited time step selector, converted from MATLAB.
"""


def compute_stable_time_step(dt: float, amr_arr, prob_def) -> float:
    """
    Return a time step that satisfies the CFL condition.

    The step is only ever reduced, never increased, from the input `dt`.

    Parameters
    ----------
    dt       : float    – candidate time step
    amr_arr  : AMRArray – current solution array (used for wave speeds)
    prob_def : object   – must expose .cfl and a callable .compute_wave_speeds(amr_arr)

    Returns
    -------
    dt_stable : float
    """
    cfl = prob_def.cfl

    wv_sp     = prob_def.compute_wave_speeds(amr_arr)
    dx        = amr_arr.dx[0]
    max_speed = float(abs(wv_sp).max())

    cfl_cur   = max_speed * dt / dx
    dt_stable = dt

    if cfl_cur > cfl:
        while cfl_cur > cfl:
            dt_stable /= 4.0
            cfl_cur    = max_speed * dt_stable / dx
        print(f" Time step reduced due to CFL — new time step {dt_stable:.6g}")

    return dt_stable

"""
run_simulation.py
Main driver script for the AMR simulation, converted from test_adaption_prob_def.m.

Usage:
    python run_simulation.py
"""

import matplotlib
matplotlib.use('TkAgg')   # or 'Qt5Agg' — change to match your environment

from ideal_gas_dynamics_1d_conservative import IdealGasDynamics1DConservative
from amr_array                          import AMRArray
from set_initial_condition              import set_initial_condition
from error_measure                      import ErrorMeasure
from perform_refinement                 import perform_refinement, assess_refinement
from time_integrator                    import TimeIntegrator
from compute_stable_time_step           import compute_stable_time_step
from plot_manager                       import PlotManager
import time as _time


def main() -> None:

    # ------------------------------------------------------------------
    # Problem definition
    # ------------------------------------------------------------------
    prob_def = IdealGasDynamics1DConservative()

    # ------------------------------------------------------------------
    # Initial AMR array and initial condition
    # ------------------------------------------------------------------
    u = AMRArray(prob_def)  # type: ignore[arg-type]
    u = set_initial_condition(u, prob_def)

    # Display initial condition
    PlotManager.plot_solution(u, 0.0, 0, u.n_coarse - 1)

    # ------------------------------------------------------------------
    # Time integrator
    # ------------------------------------------------------------------
    time_int = TimeIntegrator(prob_def)

    # Initial coarse time step (will be corrected by CFL check)
    dt = 0.001

    # ------------------------------------------------------------------
    # Time step loop
    # ------------------------------------------------------------------
    t      = 0.0
    i_step = 0

    while t < prob_def.t_end - 1.0e-8:

        # ---- Refinement pass (every n_refine steps) ------------------
        if i_step % prob_def.n_refine == 0:

            print(' performing refinement')

            em = ErrorMeasure(u, prob_def)
            em = em.analyze(u, prob_def)

            _, err_max = assess_refinement(em, prob_def.err_thr)

            print(f' err_max = {err_max:.6g}, threshold = {prob_def.err_thr:.6g}')

            u = perform_refinement(u, em, prob_def)
            u.determine_refinement_segments()

            # Rebuild sparse matrices for the updated grid
            time_int.generate_sparse_matrices(dt, u)

        # ---- CFL-limited time step -----------------------------------
        dt = compute_stable_time_step(dt, u, prob_def)

        # ---- Advance solution ----------------------------------------
        u = time_int.integrate(t, dt, u, prob_def)

        t      += dt
        i_step += 1

        print(f' i_step = {i_step}, t = {t:.6g}')

        # ---- Output (every n_out steps) ------------------------------
        if i_step % prob_def.n_out == 0 and i_step != 0:

            PlotManager.plot_solution(u, t, 0, u.n_coarse - 1)
            PlotManager.plot_solution_comparison(u, prob_def, t, 0, u.n_coarse - 1)
            PlotManager.plot_solution_levels(u, t)

            em = ErrorMeasure(u, prob_def)
            em = em.analyze(u, prob_def)
            PlotManager.plot_error_levels(em, t)

            _time.sleep(2)


if __name__ == '__main__':
    main()

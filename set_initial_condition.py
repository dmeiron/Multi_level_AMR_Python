"""
set_initial_condition.py
Staged AMR initial-condition setup, converted from MATLAB.

Builds the initial condition by gradually ramping up the amplitude over
n_iter steps, refining the mesh at each step until the error is below
the threshold.
"""

from error_measure import ErrorMeasure
from perform_refinement import perform_refinement, assess_refinement
from plot_manager import PlotManager


def set_initial_condition(obj, prob_def):
    """
    Initialise an AMRArray with the initial condition defined in prob_def,
    refining the mesh iteratively until the error indicator is below the
    threshold.

    Parameters
    ----------
    obj      : AMRArray  – pre-allocated array (modified in place; also returned)
    prob_def : object    – must expose:
                           .err_thr            – error threshold
                           .initial_condition  – callable x -> f(x)
                           .error_analyzer     – string selector for ErrorMeasure.analyze

    Returns
    -------
    obj : AMRArray – initialised (and possibly refined) array
    """
    err_thr = prob_def.err_thr
    n_iter  = 10          # number of amplitude-ramp stages

    print(f" Creating initial condition in {n_iter} steps")

    frac = 1.0 / n_iter

    for ic_iter in range(1, n_iter + 1):
        amp    = ic_iter * frac
        f_init = prob_def.initial_condition   # callable: x -> f(x)

        # Apply the full-amplitude IC, then scale down
        obj = obj.apply_function(f_init)
        obj = obj * amp

        # Measure error
        em = ErrorMeasure(obj, prob_def)
        em = em.analyze(obj, prob_def)

        refinement_required, err_max = assess_refinement(em, err_thr)
        ref_iter = 1

        print(
            f" step = {ic_iter}, ref_iter = {ref_iter}, "
            f"err_max = {err_max:.6g}, threshold = {err_thr:.6g}"
        )

        # Refine until error is below threshold
        while refinement_required:
            ref_iter += 1

            u_ref = perform_refinement(obj, em, prob_def)

            em = ErrorMeasure(u_ref, prob_def)
            em = em.analyze(u_ref, prob_def)

            refinement_required, err_max = assess_refinement(em, prob_def.err_thr)

            print(
                f" step = {ic_iter}, ref_iter = {ref_iter}, "
                f"err_max = {err_max:.6g}, threshold = {err_thr:.6g}"
            )

            obj = u_ref
            obj = obj.apply_function(f_init)
            obj = obj * amp

        PlotManager.plot_solution_levels(obj, 0)

    return obj

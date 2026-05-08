"""
perform_refinement.py
AMR refinement driver, converted from MATLAB.

Scans error indicators, copies/creates refined segments, adds guard-cell
margins, and synchronises levels.

Indexing: all 0-based.
"""

import numpy as np

from amr_array import AMRArray
from refine_from_previous_level import refine_from_previous_level


def perform_refinement(obj_to_refine: AMRArray, err_ind, prob_def) -> AMRArray:
    """
    Build a new AMRArray that is refined wherever the error indicator exceeds
    the threshold specified in prob_def.

    Parameters
    ----------
    obj_to_refine : AMRArray     – current solution array
    err_ind       : ErrorMeasure – error indicator array
    prob_def      : problem definition object
                    Must expose: err_thr, order, margin

    Returns
    -------
    obj_refined : AMRArray
    """
    verbose    = False
    err_thresh = prob_def.err_thr
    order      = prob_def.order
    margin_max = prob_def.margin

    # ----------------------------------------------------------------
    # Create refined array and copy coarse / top-level data
    # ----------------------------------------------------------------
    obj_refined = AMRArray(prob_def)
    obj_refined.x_coarse = obj_to_refine.x_coarse.copy()
    obj_refined.f_coarse = obj_to_refine.f_coarse.copy()

    n_coarse = obj_to_refine.n_coarse

    for i in range(n_coarse):
        fa = obj_to_refine.f_arr.get((i, 0))
        xc = obj_to_refine.x_coord.get((i, 0))
        if fa is not None:
            obj_refined.f_arr[(i, 0)]   = fa.copy()
        if xc is not None:
            obj_refined.x_coord[(i, 0)] = xc.copy()

    obj_refined.ref_levs_so_far = obj_to_refine.ref_levs_so_far

    # ----------------------------------------------------------------
    # Pass 1: copy or create refinements where error >= threshold
    # ----------------------------------------------------------------
    for i_coarse in range(n_coarse - 1):

        if err_ind.err_max[i_coarse, 0] < err_thresh:
            continue

        for i_lev in range(obj_to_refine.coarse_lev_depth[i_coarse]):

            if (obj_to_refine.is_ref[i_coarse, i_lev]
                    and err_ind.err_max[i_coarse, i_lev] >= err_thresh):

                # Copy the already-refined segment
                fa = obj_to_refine.f_arr.get((i_coarse, i_lev + 1))
                xc = obj_to_refine.x_coord.get((i_coarse, i_lev + 1))
                if fa is not None:
                    obj_refined.f_arr[(i_coarse, i_lev + 1)]   = fa.copy()
                if xc is not None:
                    obj_refined.x_coord[(i_coarse, i_lev + 1)] = xc.copy()

                obj_refined.is_ref[i_coarse, i_lev]    = True
                obj_refined.coarse_lev_depth[i_coarse] = i_lev + 1
                obj_refined.tot_ref_lev[i_lev + 1]    += 1

                if verbose:
                    print(f" Copying over segment at point {i_coarse} level {i_lev}")

            elif (not obj_to_refine.is_ref[i_coarse, i_lev]
                  and err_ind.err_max[i_coarse, i_lev] >= err_thresh):

                # Error too large and cell unrefined — refine now
                if verbose:
                    print(f" Refining at point {i_coarse} level {i_lev}")

                obj_refined = refine_from_previous_level(
                    obj_to_refine, i_coarse, i_lev + 1, order
                )

    # ----------------------------------------------------------------
    # Remove any empty bottom levels
    # ----------------------------------------------------------------
    lev_depth = obj_refined.ref_levs_so_far + 1
    for i_lev in range(lev_depth, -1, -1):
        if i_lev < len(obj_refined.tot_ref_lev) and obj_refined.tot_ref_lev[i_lev] == 0:
            obj_refined.delete_bottom_level()
            if verbose:
                print(f' Deleted bottom level {i_lev}')

    # ----------------------------------------------------------------
    # Identify refined segments before adding margins
    # ----------------------------------------------------------------
    obj_refined.determine_refinement_segments()

    # ----------------------------------------------------------------
    # Pass 2: add guard-cell margins around each refined segment
    # ----------------------------------------------------------------
    for i_lev in range(obj_refined.ref_levs_so_far):
        n_seg = obj_refined.n_ref_seg[i_lev]

        for i_seg in range(n_seg):
            # Variable margin: decreases at higher levels
            margin = max(margin_max - i_lev, 0)

            beg = obj_refined.beg_ref_seg[i_seg, i_lev]
            end = obj_refined.end_ref_seg[i_seg, i_lev]

            left_margin_start = max(beg - margin, 0)
            right_margin_end  = min(end + margin, obj_refined.n_coarse - 2)

            # Left margin
            for i_ref in range(left_margin_start, beg):
                if obj_to_refine.is_ref[i_ref, i_lev]:
                    if verbose:
                        print(f" Copying over left margin seg {i_seg} pt {i_ref} lev {i_lev}")
                    fa = obj_to_refine.f_arr.get((i_ref, i_lev + 1))
                    xc = obj_to_refine.x_coord.get((i_ref, i_lev + 1))
                    if fa is not None:
                        obj_refined.f_arr[(i_ref, i_lev + 1)]   = fa.copy()
                    if xc is not None:
                        obj_refined.x_coord[(i_ref, i_lev + 1)] = xc.copy()
                    obj_refined.is_ref[i_ref, i_lev]    = True
                    obj_refined.coarse_lev_depth[i_ref] = i_lev + 1
                    obj_refined.tot_ref_lev[i_lev + 1] += 1
                else:
                    if verbose:
                        print(f" Refining left margin seg {i_seg} pt {i_ref} lev {i_lev}")
                    obj_refined = refine_from_previous_level(
                        obj_refined, i_ref, i_lev + 1, order
                    )

            # Right margin
            for i_ref in range(end + 1, right_margin_end + 1):
                if obj_to_refine.is_ref[i_ref, i_lev]:
                    if verbose:
                        print(f" Copying over right margin seg {i_seg} pt {i_ref} lev {i_lev}")
                    fa = obj_to_refine.f_arr.get((i_ref, i_lev + 1))
                    xc = obj_to_refine.x_coord.get((i_ref, i_lev + 1))
                    if fa is not None:
                        obj_refined.f_arr[(i_ref, i_lev + 1)]   = fa.copy()
                    if xc is not None:
                        obj_refined.x_coord[(i_ref, i_lev + 1)] = xc.copy()
                    obj_refined.is_ref[i_ref, i_lev]    = obj_to_refine.is_ref[i_ref, i_lev]
                    obj_refined.coarse_lev_depth[i_ref] = i_lev + 1
                    obj_refined.tot_ref_lev[i_lev + 1] += 1
                else:
                    if verbose:
                        print(f" Refining right margin seg {i_seg} pt {i_ref} lev {i_lev}")
                    obj_refined = refine_from_previous_level(
                        obj_refined, i_ref, i_lev + 1, order
                    )

    # ----------------------------------------------------------------
    # Final: recompute segment locations and synchronise levels
    # ----------------------------------------------------------------
    obj_refined.determine_refinement_segments()
    obj_refined.fine_to_coarse()

    return obj_refined


def assess_refinement(em, err_thr: float):
    """
    Inspect an ErrorMeasure and decide whether further refinement is needed.

    Parameters
    ----------
    em      : ErrorMeasure
    err_thr : float

    Returns
    -------
    refinement_required : bool
    err_max             : float  – global maximum error indicator
    """
    err_max = float(np.max(em.err_max))
    return err_max >= err_thr, err_max

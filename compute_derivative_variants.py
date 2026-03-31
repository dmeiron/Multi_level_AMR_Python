"""
compute_derivative_variants.py
Single-level and no-sync derivative helpers, converted from MATLAB.

Both functions follow the same persistent-cache pattern as compute_derivative.py.
They share the same module-level FDDeriv cache so the object is only
constructed once regardless of which function is called first.

All indexing is 0-based.
"""

import copy as _copy
import numpy as np
from fd_deriv import FDDeriv

# Shared module-level cache: order -> FDDeriv instance
_der_cache: dict[int, FDDeriv] = {}


def _get_der(order: int) -> FDDeriv:
    """Return (and cache) a FDDeriv instance for the given order."""
    if order not in _der_cache:
        _der_cache[order] = FDDeriv(order)
    return _der_cache[order]


# ---------------------------------------------------------------------------
# compute_derivative_at_level
# ---------------------------------------------------------------------------

def compute_derivative_at_level(obj, level: int, order: int, n_deriv: int):
    """
    Differentiate a single refinement level of an AMRArray.

    Parameters
    ----------
    obj     : AMRArray  – source array (not modified)
    level   : int       – 0-indexed refinement level to differentiate
    order   : int       – finite-difference order
    n_deriv : int       – 1 (first) or 2 (second) derivative

    Returns
    -------
    obj_deriv : AMRArray – copy with derivatives stored at `level`
    """
    der       = _get_der(order)
    obj_deriv = _copy.deepcopy(obj)
    n_comp    = obj.n_comp
    n_seg     = obj.n_ref_seg[level - 1]

    der.normalize_derivatives(obj.dx[level])

    for i_seg in range(n_seg):
        i_start = obj.beg_ref_seg[i_seg, level - 1]
        i_end   = obj.end_ref_seg[i_seg, level - 1]

        for i_comp in range(n_comp):
            values   = obj.get_contiguous_refinement_array(level, i_comp, i_start, i_end)
            n_values = len(values)

            if n_deriv == 1:
                d_values = der.compute_first_derivative(n_values, values)
            elif n_deriv == 2:
                d_values = der.compute_second_derivative(n_values, values)
            else:
                raise ValueError("n_deriv must be 1 or 2.")

            obj_deriv.set_contiguous_refinement_array(
                level, i_comp, i_start, i_end, d_values
            )

    _der_cache[order] = der
    return obj_deriv


# ---------------------------------------------------------------------------
# compute_derivative_no_sync
# ---------------------------------------------------------------------------

def compute_derivative_no_sync(obj, order: int, n_deriv: int):
    """
    Differentiate all levels of an AMRArray without synchronising
    (no fine_to_coarse call at the end).

    Used by the error indicator (error_measure.func_var) where synchronisation
    would corrupt the error estimate.

    Parameters
    ----------
    obj     : AMRArray  – source array (not modified)
    order   : int       – finite-difference order
    n_deriv : int       – 1 (first) or 2 (second) derivative

    Returns
    -------
    obj_deriv : AMRArray
    """
    der       = _get_der(order)
    obj_deriv = _copy.deepcopy(obj)
    n_comp    = obj.n_comp

    if n_deriv == 1:

        # --- Coarse level (level 0) ---
        der.normalize_derivatives(obj.dx[0])
        n_values = obj.n_coarse

        for i_comp in range(n_comp):
            values   = obj.f_coarse[:, i_comp]
            d_values = der.compute_first_derivative(n_values, values)
            obj_deriv.f_coarse[:, i_comp] = d_values
            obj_deriv.set_contiguous_refinement_array(
                0, i_comp, 0, n_values - 2, d_values
            )

        # --- Refined levels ---
        for i_lev in range(1, obj.ref_levs_so_far + 1):
            n_seg = obj.n_ref_seg[i_lev - 1]
            der.normalize_derivatives(obj.dx[i_lev])

            for i_seg in range(n_seg):
                i_start = obj.beg_ref_seg[i_seg, i_lev - 1]
                i_end   = obj.end_ref_seg[i_seg, i_lev - 1]

                for i_comp in range(n_comp):
                    values   = obj.get_contiguous_refinement_array(
                        i_lev, i_comp, i_start, i_end
                    )
                    n_values = len(values)
                    d_values = der.compute_first_derivative(n_values, values)
                    obj_deriv.set_contiguous_refinement_array(
                        i_lev, i_comp, i_start, i_end, d_values
                    )

    elif n_deriv == 2:

        # --- Coarse level (level 0) ---
        der.normalize_derivatives(obj.dx[0])
        n_values = obj.n_coarse

        for i_comp in range(n_comp):
            values   = obj.f_coarse[:, i_comp]
            d_values = der.compute_second_derivative(n_values, values)
            obj_deriv.f_coarse[:, i_comp] = d_values
            obj_deriv.set_contiguous_refinement_array(
                0, i_comp, 0, n_values - 2, d_values
            )

        # --- Refined levels ---
        for i_lev in range(1, obj.ref_levs_so_far + 1):
            n_seg = obj.n_ref_seg[i_lev - 1]
            der.normalize_derivatives(obj.dx[i_lev])

            for i_seg in range(n_seg):
                i_start = obj.beg_ref_seg[i_seg, i_lev - 1]
                i_end   = obj.end_ref_seg[i_seg, i_lev - 1]

                for i_comp in range(n_comp):
                    values   = obj.get_contiguous_refinement_array(
                        i_lev, i_comp, i_start, i_end
                    )
                    n_values = len(values)
                    d_values = der.compute_second_derivative(n_values, values)
                    obj_deriv.set_contiguous_refinement_array(
                        i_lev, i_comp, i_start, i_end, d_values
                    )

    else:
        raise ValueError("n_deriv must be 1 or 2.")

    _der_cache[order] = der
    return obj_deriv


def reset_derivative_variant_cache():
    """Clear the shared FDDeriv cache (useful for testing)."""
    global _der_cache
    _der_cache.clear()

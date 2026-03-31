"""
compute_derivative.py
Standalone function to differentiate an AMRArray, converted from MATLAB.

The MATLAB version used a `persistent` variable to cache the FDDeriv object
across calls.  In Python this is handled with a module-level singleton dict
so that the cached object survives multiple calls within a session but is
easy to reset if needed.

Usage
-----
    from compute_derivative import compute_derivative

    obj_deriv = compute_derivative(amr_obj, order=4, n_deriv=1)

Notes on indexing
-----------------
All indices follow the 0-based convention used in amr_array.py and
fd_deriv.py.  In particular:
  - levels     : 0 = coarse, 1 = first refinement, ...
  - components : 0-indexed
  - beg/end_ref_seg arrays : 0-indexed
"""

import numpy as np
from fd_deriv import FDDeriv

# Module-level cache: maps `order` -> FDDeriv instance.
# Mimics MATLAB's `persistent` behaviour.
_der_cache: dict[int, FDDeriv] = {}


def compute_derivative(obj, order: int, n_deriv: int):
    """
    Differentiate every level of an AMRArray.

    Parameters
    ----------
    obj     : AMRArray  – the array to differentiate (not modified in place)
    order   : int       – finite-difference order of accuracy
    n_deriv : int       – 1 for first derivative, 2 for second derivative

    Returns
    -------
    obj_deriv : AMRArray  – a new AMRArray containing the derivatives
    """
    global _der_cache

    # Retrieve or construct the cached FDDeriv for this order
    if order not in _der_cache:
        _der_cache[order] = FDDeriv(order)
    der = _der_cache[order]

    # Work on a copy so the original is unchanged
    obj_deriv = _copy_amr(obj)

    n_comp = obj.n_comp

    if n_deriv == 1:
        # ------------------------------------------------------------------
        # First derivative
        # ------------------------------------------------------------------

        # --- Coarse level (level 0) ---------------------------------------
        der.normalize_derivatives(obj.dx[0])
        n_values = obj.n_coarse

        for i_comp in range(n_comp):
            values   = obj.f_coarse[:, i_comp]
            d_values = der.compute_first_derivative(n_values, values)
            obj_deriv.f_coarse[:, i_comp] = d_values
            obj_deriv.set_contiguous_refinement_array(
                level=0, comp=i_comp,
                index_start=0, index_end=n_values - 2,
                values=d_values,
            )

        # --- Refined levels (levels 1 .. ref_levs_so_far) -----------------
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
        # ------------------------------------------------------------------
        # Second derivative
        # ------------------------------------------------------------------

        # --- Coarse level (level 0) ---------------------------------------
        der.normalize_derivatives(obj.dx[0])
        n_values = obj.n_coarse

        for i_comp in range(n_comp):
            values   = obj.f_coarse[:, i_comp]
            d_values = der.compute_second_derivative(n_values, values)
            obj_deriv.f_coarse[:, i_comp] = d_values
            obj_deriv.set_contiguous_refinement_array(
                level=0, comp=i_comp,
                index_start=0, index_end=n_values - 2,
                values=d_values,
            )

        # --- Refined levels -----------------------------------------------
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
        raise ValueError("n_deriv must be 1 (first) or 2 (second derivative).")

    # Synchronise refined levels back to coarser ones
    obj_deriv.fine_to_coarse()

    # Update the cache with the (possibly normalised) der object
    _der_cache[order] = der

    return obj_deriv


def reset_derivative_cache():
    """
    Clear the module-level FDDeriv cache.
    Useful in tests or when order needs to change mid-run.
    """
    global _der_cache
    _der_cache.clear()


# ------------------------------------------------------------------
# Helper: shallow structural copy of an AMRArray
# ------------------------------------------------------------------

import copy as _copy

def _copy_amr(obj):
    """Return a deep copy of an AMRArray (delegates to copy.deepcopy)."""
    return _copy.deepcopy(obj)

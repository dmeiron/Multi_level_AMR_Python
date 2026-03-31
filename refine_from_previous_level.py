"""
refine_from_previous_level.py
Lagrange-interpolation-based refinement helper, converted from MATLAB.

Indexing: all 0-based.

The MATLAB `persistent li` is replaced by a module-level cache keyed by order.
"""

import numpy as np
from lagrange_interp import LagrangeInterp

# Module-level cache: order -> LagrangeInterp instance
_li_cache: dict[int, LagrangeInterp] = {}


def refine_from_previous_level(obj, index: int, level: int, order: int):
    """
    Refine the AMRArray `obj` at coarse index `index` to refinement `level`
    using Lagrange interpolation of the given `order`.

    Parameters
    ----------
    obj   : AMRArray  – array to refine (mutated in place; also returned)
    index : int       – 0-indexed coarse grid point
    level : int       – target refinement level (0-indexed; 0 = coarse)
    order : int       – Lagrange interpolation order

    Returns
    -------
    obj : AMRArray (same object, mutated)
    """
    global _li_cache

    if level >= obj.max_no_levs:
        raise ValueError("Refinement level exceeds maximum number of levels.")

    if order not in _li_cache:
        _li_cache[order] = LagrangeInterp(order)
    li = _li_cache[order]

    n_comp = obj.n_comp
    n_ref  = obj.n_ref[level]

    # ----------------------------------------------------------------
    # Gather source x-coordinates and function values from level-1
    # ----------------------------------------------------------------
    if level == 1:
        # Source is the coarse grid (level 0)
        nc = obj.n_coarse
        x = np.zeros(nc)
        f = np.zeros((nc, n_comp))

        for i in range(nc - 1):
            xc = obj.x_coord.get((i, 0))
            fa = obj.f_arr.get((i, 0))
            if xc is not None:
                x[i] = xc[0]
            if fa is not None:
                f[i, :n_comp] = fa[0, :n_comp]

        # Right endpoint of last coarse interval
        xc_last = obj.x_coord.get((nc - 2, 0))
        if xc_last is not None:
            x[nc - 1] = xc_last[1]
    else:
        # Source is level-1 data at this index
        x = obj.x_coord.get((index, level - 1), None)
        f = obj.f_arr.get((index, level - 1), None)
        if x is None or f is None:
            raise RuntimeError(
                f"No data at (index={index}, level={level-1}) to refine from."
            )

    # ----------------------------------------------------------------
    # Interpolate to produce the refined arrays
    # ----------------------------------------------------------------
    x_ref = np.zeros(n_ref + 1)
    f_ref = np.zeros((n_ref + 1, n_comp))

    if level == 1:
        # Interpolate a single coarse interval `index` to n_ref sub-points
        for i_comp in range(n_comp):
            x_interp, f_interp = li.interpolate_at_reg_spaced_pts(
                x, f[:, i_comp], index, n_ref
            )
            x_ref[:] = x_interp
            f_ref[:, i_comp] = f_interp

    else:
        # Each sub-interval of the level-(level-1) segment is further split
        # by ref_fac.  n_ref(level-1) sub-intervals exist.
        n_r_prev = obj.n_ref[level - 1]
        rf       = obj.ref_fac

        for i_comp in range(n_comp):
            for i_int in range(n_r_prev):   # 0-indexed sub-interval
                x_interp, f_interp = li.interpolate_at_reg_spaced_pts(
                    x, f[:, i_comp], i_int, rf
                )
                for j in range(rf + 1):
                    x_ref[rf * i_int + j]         = x_interp[j]
                    f_ref[rf * i_int + j, i_comp] = f_interp[j]

    # ----------------------------------------------------------------
    # Store results
    # ----------------------------------------------------------------
    obj.x_coord[(index, level)] = x_ref
    obj = obj.set_refinement_at(index, level, f_ref)

    return obj


def reset_refine_cache():
    """Clear the LagrangeInterp cache (useful for testing)."""
    global _li_cache
    _li_cache.clear()

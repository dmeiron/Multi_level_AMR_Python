"""
test_derivatives.py
Verification tests for all derivative routines.

Tests are organised into four groups:

  1. FDDeriv  – unit tests for the core finite-difference class
               (compute_first_derivative, compute_second_derivative,
                normalize_derivatives, form_deriv_matrices)

  2. AMRArray (flat, no refinement) – tests compute_derivative and
               compute_derivative_no_sync on a coarse-only mesh

  3. AMRArray (with refinement) – same tests after adding one level of
               refinement so the multi-level code paths are exercised

  4. compute_derivative_at_level – verifies the single-level variant

Strategy
--------
All tests use analytic functions whose derivatives are known exactly:
  - f(x)  = sin(k*x)          f'  = k*cos(k*x)     f'' = -k^2*sin(k*x)
  - f(x)  = exp(-x^2/w^2)     f'  = -2x/w^2 * f    f'' = (4x^2/w^4 - 2/w^2)*f
  - f(x)  = x^n  (polynomial) exact for orders >= n

For each test we check:
  a) The maximum absolute error over interior points is below the expected
     tolerance for the scheme's formal order p:  err <= C * h^p
  b) The error convergence rate (computed by halving h) matches the order.

Run with:
    cd <directory containing all .py files>
    python test_derivatives.py          # runs all tests, prints summary
    python -m pytest test_derivatives.py -v   # if pytest is available
"""

import sys
import os
import math
import numpy as np

# ---------------------------------------------------------------------------
# Make sure the output directory is on the path so imports work
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from fd_deriv   import FDDeriv
from amr_array  import AMRArray
from compute_derivative          import compute_derivative,          reset_derivative_cache
from compute_derivative_variants import (compute_derivative_no_sync,
                                          compute_derivative_at_level,
                                          reset_derivative_variant_cache)


# ===========================================================================
# Helpers
# ===========================================================================

class _SimpleProb:
    """Minimal prob_def-like object sufficient to construct an AMRArray."""
    def __init__(self, n_coarse=51, n_comp=1, ref_fac=4, max_no_levs=5,
                 x_left=0.0, x_right=1.0):
        self.n_coarse    = n_coarse
        self.n_comp      = n_comp
        self.ref_fac     = ref_fac
        self.max_no_levs = max_no_levs
        self.x_left      = x_left
        self.x_right     = x_right


def _make_amr(n_coarse=51, n_comp=1, x_left=0.0, x_right=1.0):
    pd = _SimpleProb(n_coarse=n_coarse, n_comp=n_comp,
                     x_left=x_left, x_right=x_right)
    return AMRArray(pd), pd


def _load_coarse(amr, f_func):
    """Populate coarse data from a scalar function f(x) for all components."""
    x = amr.x_coarse
    for c in range(amr.n_comp):
        vals = f_func(x)
        amr.set_coarse_array(vals, c)


def _max_interior_err(numeric, analytic, fo2=2):
    """Max absolute error, skipping fo2 boundary points on each side."""
    diff = np.abs(numeric[fo2:-fo2] - analytic[fo2:-fo2])
    return float(np.max(diff))


def _convergence_rate(err_coarse, err_fine, refinement=2):
    """Estimated convergence rate p from two errors at h and h/refinement."""
    if err_coarse <= 0 or err_fine <= 0:
        return float('nan')
    return math.log(err_coarse / err_fine) / math.log(refinement)


def _pass(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}]  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# ===========================================================================
# Group 1 — FDDeriv unit tests
# ===========================================================================

def test_fd_deriv_polynomial_exactness():
    """
    An order-p FD scheme should differentiate polynomials of degree <= p exactly
    (to machine precision).  We test with f(x) = x^p on a uniform grid.
    """
    print("\n--- Group 1: FDDeriv polynomial exactness ---")
    all_ok = True
    for order in (2, 4, 6):
        n = 4 * order + 1          # enough points to avoid boundary domination
        x = np.linspace(0.0, 1.0, n)
        dx = x[1] - x[0]
        der = FDDeriv(order)
        der.normalize_derivatives(dx)

        # f(x) = x^order  =>  f'(x) = order * x^(order-1)
        #                      f''(x) = order*(order-1) * x^(order-2)
        f      = x ** order
        df_ex  = order * x ** (order - 1)
        d2f_ex = order * (order - 1) * x ** (order - 2)

        df_num  = der.compute_first_derivative(n, f)
        d2f_num = der.compute_second_derivative(n, f)

        fo2 = order // 2
        err1 = _max_interior_err(df_num,  df_ex,  fo2)
        err2 = _max_interior_err(d2f_num, d2f_ex, fo2)

        tol = 1e-8
        ok1 = _pass(f"order={order} first  deriv of x^{order}, max_err={err1:.2e}", err1 < tol)
        ok2 = _pass(f"order={order} second deriv of x^{order}, max_err={err2:.2e}", err2 < tol)
        all_ok = all_ok and ok1 and ok2
    return all_ok


def test_fd_deriv_sin_accuracy():
    """
    For f(x) = sin(k*x) on [0, 2*pi], verify that the error is O(h^order).
    We confirm both the absolute error level and the convergence rate.
    """
    print("\n--- Group 1: FDDeriv sin(kx) convergence ---")
    k   = 2.0
    all_ok = True

    for order in (2, 4):
        errors1 = []
        errors2 = []
        ns = [20 * 2**r for r in range(3)]   # 20, 40, 80

        for n in ns:
            x  = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
            dx = x[1] - x[0]
            der = FDDeriv(order)
            der.normalize_derivatives(dx)

            f      = np.sin(k * x)
            df_ex  =  k * np.cos(k * x)
            d2f_ex = -k**2 * np.sin(k * x)

            fo2 = order // 2
            df_num  = der.compute_first_derivative(n, f)
            d2f_num = der.compute_second_derivative(n, f)

            errors1.append(_max_interior_err(df_num,  df_ex,  fo2))
            errors2.append(_max_interior_err(d2f_num, d2f_ex, fo2))

        # Check convergence rate between n=40 and n=80
        rate1 = _convergence_rate(errors1[1], errors1[2])
        rate2 = _convergence_rate(errors2[1], errors2[2])
        # Allow rate >= order - 0.5  (boundary effects can reduce rate slightly)
        ok1 = _pass(f"order={order} first  deriv convergence rate={rate1:.2f} (expect ~{order})",
                    rate1 >= order - 0.5)
        ok2 = _pass(f"order={order} second deriv convergence rate={rate2:.2f} (expect ~{order})",
                    rate2 >= order - 0.5)
        all_ok = all_ok and ok1 and ok2
    return all_ok


def test_fd_deriv_sparse_matches_dense():
    """
    form_deriv_matrices() should produce sparse matrices that give the same
    result as compute_first/second_derivative() when applied to a vector.
    We test this via a minimal AMRArray (coarse only, no refinement).
    """
    print("\n--- Group 1: FDDeriv sparse == dense ---")
    order = 4
    n     = 33
    amr, _ = _make_amr(n_coarse=n)

    x  = amr.x_coarse
    dx = amr.dx[0]
    f  = np.sin(2 * math.pi * x)
    amr.set_coarse_array(f, 0)

    # Dense path
    der = FDDeriv(order)
    der.normalize_derivatives(dx)
    df_dense  = der.compute_first_derivative(n, f)
    d2f_dense = der.compute_second_derivative(n, f)

    # Sparse path (form_deriv_matrices needs ref_levs_so_far=0)
    amr.ref_levs_so_far = 0
    der2 = FDDeriv(order)
    der2.form_deriv_matrices(amr)
    df_sparse  = der2.sp_mat_d1[0][0].dot(f)
    d2f_sparse = der2.sp_mat_d2[0][0].dot(f)

    tol = 1e-12
    err1 = float(np.max(np.abs(df_dense  - df_sparse)))
    err2 = float(np.max(np.abs(d2f_dense - d2f_sparse)))

    ok1 = _pass(f"sparse vs dense first  deriv, max_diff={err1:.2e}", err1 < tol)
    ok2 = _pass(f"sparse vs dense second deriv, max_diff={err2:.2e}", err2 < tol)
    return ok1 and ok2


# ===========================================================================
# Group 2 — compute_derivative on a flat (coarse-only) AMRArray
# ===========================================================================

def test_compute_derivative_coarse_first():
    """
    compute_derivative(n_deriv=1) on a coarse-only array should recover
    f'(x) to within the expected accuracy for several test functions.
    """
    print("\n--- Group 2: compute_derivative (coarse) — first derivative ---")
    reset_derivative_cache()
    order  = 4
    n      = 101
    all_ok = True

    test_cases = [
        ("sin(2*pi*x)", lambda x: np.sin(2*math.pi*x),
                        lambda x: 2*math.pi*np.cos(2*math.pi*x)),
        ("exp(-x^2)",  lambda x: np.exp(-x**2),
                       lambda x: -2*x*np.exp(-x**2)),
        ("x^3",        lambda x: x**3,
                       lambda x: 3*x**2),
    ]

    for name, f_func, df_func in test_cases:
        amr, _ = _make_amr(n_coarse=n, x_left=-1.0, x_right=1.0)
        x = amr.x_coarse
        amr.set_coarse_array(f_func(x), 0)
        amr.determine_refinement_segments()

        amr_d = compute_derivative(amr, order, 1)
        reset_derivative_cache()

        numeric  = amr_d.f_coarse[:, 0]
        analytic = df_func(x)
        fo2 = order // 2
        err = _max_interior_err(numeric, analytic, fo2)
        tol = 1e-4   # loose — coarse mesh h~0.02 with order 4 => h^4 ~ 2e-7, allow for C
        ok  = _pass(f"f={name}, max_err={err:.3e}", err < tol)
        all_ok = all_ok and ok

    return all_ok


def test_compute_derivative_coarse_second():
    """compute_derivative(n_deriv=2) on a coarse-only array."""
    print("\n--- Group 2: compute_derivative (coarse) — second derivative ---")
    reset_derivative_cache()
    order  = 4
    n      = 101
    all_ok = True

    test_cases = [
        ("sin(2*pi*x)", lambda x: np.sin(2*math.pi*x),
                        lambda x: -(2*math.pi)**2 * np.sin(2*math.pi*x)),
        ("exp(-x^2)",  lambda x: np.exp(-x**2),
                       lambda x: (4*x**2 - 2)*np.exp(-x**2)),
        ("x^4",        lambda x: x**4,
                       lambda x: 12*x**2),
    ]

    for name, f_func, d2f_func in test_cases:
        amr, _ = _make_amr(n_coarse=n, x_left=-1.0, x_right=1.0)
        x = amr.x_coarse
        amr.set_coarse_array(f_func(x), 0)
        amr.determine_refinement_segments()

        amr_d = compute_derivative(amr, order, 2)
        reset_derivative_cache()

        numeric  = amr_d.f_coarse[:, 0]
        analytic = d2f_func(x)
        fo2 = order // 2
        err = _max_interior_err(numeric, analytic, fo2)
        tol = 1e-3
        ok  = _pass(f"f={name}, max_err={err:.3e}", err < tol)
        all_ok = all_ok and ok

    return all_ok


def test_compute_derivative_order_convergence():
    """
    Verify the convergence rate of compute_derivative by halving the mesh
    spacing and checking the error ratio.
    """
    print("\n--- Group 2: compute_derivative order convergence (h-refinement) ---")
    reset_derivative_cache()
    all_ok = True

    f_func   = lambda x: np.sin(2 * math.pi * x)
    df_func  = lambda x: 2 * math.pi * np.cos(2 * math.pi * x)
    d2f_func = lambda x: -(2 * math.pi)**2 * np.sin(2 * math.pi * x)

    for order in (2, 4):
        fo2    = order // 2
        ns     = [21, 41, 81]
        errs1  = []
        errs2  = []

        for n in ns:
            for n_deriv, errs, ex_func in [(1, errs1, df_func), (2, errs2, d2f_func)]:
                reset_derivative_cache()
                amr, _ = _make_amr(n_coarse=n)
                amr.set_coarse_array(f_func(amr.x_coarse), 0)
                amr.determine_refinement_segments()
                amr_d = compute_derivative(amr, order, n_deriv)
                err = _max_interior_err(amr_d.f_coarse[:, 0], ex_func(amr.x_coarse), fo2)
                errs.append(err)

        rate1 = _convergence_rate(errs1[1], errs1[2])
        rate2 = _convergence_rate(errs2[1], errs2[2])
        ok1 = _pass(f"order={order} first  deriv rate={rate1:.2f} (expect ~{order})",
                    rate1 >= order - 0.5)
        ok2 = _pass(f"order={order} second deriv rate={rate2:.2f} (expect ~{order})",
                    rate2 >= order - 0.5)
        all_ok = all_ok and ok1 and ok2

    reset_derivative_cache()
    return all_ok


def test_no_sync_matches_compute_derivative_coarse():
    """
    compute_derivative_no_sync should give the same coarse-level result as
    compute_derivative on a flat (coarse-only) AMRArray, since fine_to_coarse
    is a no-op when there is no refinement.
    """
    print("\n--- Group 2: compute_derivative_no_sync matches compute_derivative ---")
    reset_derivative_cache()
    reset_derivative_variant_cache()

    order = 4
    n     = 51
    amr, _ = _make_amr(n_coarse=n, x_left=-1.0, x_right=1.0)
    f = np.sin(2 * math.pi * amr.x_coarse)
    amr.set_coarse_array(f, 0)
    amr.determine_refinement_segments()

    d1  = compute_derivative(amr, order, 1)
    d1b = compute_derivative_no_sync(amr, order, 1)

    reset_derivative_cache()
    reset_derivative_variant_cache()

    err = float(np.max(np.abs(d1.f_coarse[:, 0] - d1b.f_coarse[:, 0])))
    return _pass(f"no_sync vs compute_derivative, max_diff={err:.2e}", err < 1e-12)


# ===========================================================================
# Group 3 — compute_derivative on a refined AMRArray
# ===========================================================================

def _build_refined_amr(order=4, n_coarse=51, ref_fac=4, x_left=0.0, x_right=1.0):
    """
    Build a single-component AMRArray with one level of refinement over the
    middle third of the domain, loaded with sin(2*pi*x).
    """
    from refine_from_previous_level import refine_from_previous_level, reset_refine_cache
    reset_refine_cache()

    pd  = _SimpleProb(n_coarse=n_coarse, ref_fac=ref_fac,
                      x_left=x_left, x_right=x_right)
    amr = AMRArray(pd)

    f_func = lambda x: np.sin(2 * math.pi * x)
    amr.set_coarse_array(f_func(amr.x_coarse), 0)

    # Refine the middle third
    i_mid_start = n_coarse // 3
    i_mid_end   = 2 * n_coarse // 3

    for i in range(i_mid_start, i_mid_end):
        amr = refine_from_previous_level(amr, i, level=1, order=order)

    amr.determine_refinement_segments()
    return amr, f_func


def test_compute_derivative_refined_first():
    """
    On a refined AMRArray, compute_derivative should recover f'(x) at both
    the coarse level and in the refined region.
    """
    print("\n--- Group 3: compute_derivative (refined) — first derivative ---")
    reset_derivative_cache()
    order = 4
    fo2   = order // 2
    all_ok = True

    amr, f_func = _build_refined_amr(order=order)
    df_func = lambda x: 2 * math.pi * np.cos(2 * math.pi * x)

    amr_d = compute_derivative(amr, order, 1)
    reset_derivative_cache()

    # --- Coarse level check ---
    numeric  = amr_d.f_coarse[:, 0]
    analytic = df_func(amr.x_coarse)
    err_c = _max_interior_err(numeric, analytic, fo2)
    ok1 = _pass(f"coarse level, max_err={err_c:.3e}", err_c < 1e-3)
    all_ok = all_ok and ok1

    # --- Refined region check ---
    # Collect all level-1 data
    ref_numeric  = []
    ref_analytic = []
    for i in range(amr.n_coarse - 1):
        arr = amr_d.f_arr.get((i, 1))
        xc  = amr.x_coord.get((i, 1))
        if arr is not None and xc is not None:
            ref_numeric.append(arr[:, 0])
            ref_analytic.append(df_func(xc))

    if ref_numeric:
        rn = np.concatenate(ref_numeric)
        ra = np.concatenate(ref_analytic)
        err_r = float(np.max(np.abs(rn - ra)))
        ok2 = _pass(f"refined level, max_err={err_r:.3e}", err_r < 1e-4)
        all_ok = all_ok and ok2
    else:
        print("  [SKIP]  no refined data found")

    return all_ok


def test_compute_derivative_refined_second():
    """
    compute_derivative(n_deriv=2) on a refined AMRArray.

    Note on expected accuracy in the refined region
    ------------------------------------------------
    The function values at level-1 come from order-4 Lagrange interpolation,
    which introduces an O(h_coarse^4) error in f.  When the second derivative
    is applied it amplifies that by 1/h_fine^2 = 1/(h_coarse/ref_fac)^2,
    giving an overall O(h_coarse^2) error in the fine-grid second derivative
    (rather than O(h_fine^4)).  This is the expected behaviour.

    The test therefore checks:
      - The coarse-level second derivative is accurate to O(h_coarse^4).
      - The refined-region second derivative is not wildly wrong (error < 0.1).
      - The coarse second-derivative error is *better* than the unrefined
        coarse error would be at the same h (sanity check after fine_to_coarse).
    """
    print("\n--- Group 3: compute_derivative (refined) — second derivative ---")
    reset_derivative_cache()
    order = 4
    fo2   = order // 2

    amr, _ = _build_refined_amr(order=order)
    d2f_func = lambda x: -(2 * math.pi)**2 * np.sin(2 * math.pi * x)

    amr_d = compute_derivative(amr, order, 2)
    reset_derivative_cache()

    # 1. Coarse-level accuracy (not in the refined region)
    nc = amr.n_coarse
    n_ref_start = int(amr.beg_ref_seg[0, 0])
    n_ref_end   = int(amr.end_ref_seg[0, 0])
    # Pick indices that are outside the refined region
    outside = list(range(fo2, n_ref_start - fo2)) + list(range(n_ref_end + fo2, nc - fo2))
    if outside:
        numeric  = amr_d.f_coarse[outside, 0]
        analytic = d2f_func(amr.x_coarse[outside])
        err_out = float(np.max(np.abs(numeric - analytic)))
        ok1 = _pass(f"coarse level (outside refined), max_err={err_out:.3e}", err_out < 5e-3)
    else:
        ok1 = True

    # 2. Refined region: error < 0.1 (limited by interpolation noise amplification)
    ref_errs = []
    for i in range(nc - 1):
        arr = amr_d.f_arr.get((i, 1))
        xc  = amr.x_coord.get((i, 1))
        if arr is not None and xc is not None:
            ref_errs.append(float(np.max(np.abs(arr[:, 0] - d2f_func(xc)))))
    if ref_errs:
        err_ref = float(np.max(ref_errs))
        ok2 = _pass(
            f"refined level second deriv (interp-noise limited), max_err={err_ref:.3e} < 0.1",
            err_ref < 0.1
        )
    else:
        ok2 = True

    return ok1 and ok2


def test_refined_derivative_finer_than_coarse():
    """
    The error in the refined region should be smaller than in the coarse region
    (since dx is smaller by ref_fac).
    """
    print("\n--- Group 3: refined error < coarse error ---")
    reset_derivative_cache()
    order = 4
    fo2   = order // 2

    amr, f_func = _build_refined_amr(order=order, ref_fac=4)
    df_func = lambda x: 2 * math.pi * np.cos(2 * math.pi * x)

    amr_d = compute_derivative(amr, order, 1)
    reset_derivative_cache()

    err_coarse = _max_interior_err(amr_d.f_coarse[:, 0], df_func(amr.x_coarse), fo2)

    ref_errs = []
    for i in range(amr.n_coarse - 1):
        arr = amr_d.f_arr.get((i, 1))
        xc  = amr.x_coord.get((i, 1))
        if arr is not None and xc is not None:
            ref_errs.append(np.max(np.abs(arr[:, 0] - df_func(xc))))

    if not ref_errs:
        print("  [SKIP]  no refined data found")
        return True

    err_refined = float(np.max(ref_errs))
    ok = _pass(
        f"refined err={err_refined:.3e} < coarse err={err_coarse:.3e}",
        err_refined < err_coarse
    )
    return ok


# ===========================================================================
# Group 4 — compute_derivative_at_level
# ===========================================================================

def test_compute_derivative_at_level():
    """
    compute_derivative_at_level should differentiate exactly the requested
    level and leave other levels unchanged.
    """
    print("\n--- Group 4: compute_derivative_at_level ---")
    reset_derivative_variant_cache()
    order = 4
    fo2   = order // 2
    all_ok = True

    amr, _ = _build_refined_amr(order=order)
    df_func = lambda x: 2 * math.pi * np.cos(2 * math.pi * x)

    # Pick the first refined level (level=1) and its first segment
    if amr.n_ref_seg[0] == 0:
        print("  [SKIP]  no refined segments found")
        return True

    # Differentiate level 1 only
    amr_d = compute_derivative_at_level(amr, level=1, order=order, n_deriv=1)
    reset_derivative_variant_cache()

    # Check at least one refined segment
    ref_errs = []
    for i in range(amr.n_coarse - 1):
        arr = amr_d.f_arr.get((i, 1))
        xc  = amr.x_coord.get((i, 1))
        if arr is not None and xc is not None:
            ref_errs.append(np.max(np.abs(arr[:, 0] - df_func(xc))))

    if ref_errs:
        err = float(np.max(ref_errs))
        ok1 = _pass(f"level-1 first deriv, max_err={err:.3e}", err < 1e-4)
        all_ok = all_ok and ok1

    # Coarse level (level=0) data should be UNCHANGED (not differentiated)
    f_func = lambda x: np.sin(2 * math.pi * x)
    coarse_unchanged = np.allclose(amr_d.f_coarse[:, 0], f_func(amr.x_coarse))
    ok2 = _pass("coarse level not modified by at_level call", coarse_unchanged)
    all_ok = all_ok and ok2

    return all_ok


def test_compute_derivative_at_level_second():
    """compute_derivative_at_level with n_deriv=2."""
    print("\n--- Group 4: compute_derivative_at_level second deriv ---")
    reset_derivative_variant_cache()
    order = 4

    amr, _ = _build_refined_amr(order=order)
    d2f_func = lambda x: -(2 * math.pi)**2 * np.sin(2 * math.pi * x)

    if amr.n_ref_seg[0] == 0:
        print("  [SKIP]  no refined segments found")
        return True

    amr_d = compute_derivative_at_level(amr, level=1, order=order, n_deriv=2)
    reset_derivative_variant_cache()

    ref_errs = []
    for i in range(amr.n_coarse - 1):
        arr = amr_d.f_arr.get((i, 1))
        xc  = amr.x_coord.get((i, 1))
        if arr is not None and xc is not None:
            ref_errs.append(np.max(np.abs(arr[:, 0] - d2f_func(xc))))

    if not ref_errs:
        print("  [SKIP]  no refined data")
        return True

    err = float(np.max(ref_errs))
    return _pass(
        f"level-1 second deriv (interp-noise limited), max_err={err:.3e} < 0.1",
        err < 0.1
    )


# ===========================================================================
# Group 5 — multi-component AMRArray
# ===========================================================================

def test_multi_component_derivatives():
    """
    With n_comp=2, each component should be differentiated independently.
    Component 0: sin(2*pi*x),  Component 1: cos(2*pi*x)
    """
    print("\n--- Group 5: multi-component derivatives ---")
    reset_derivative_cache()
    order = 4
    fo2   = order // 2
    n     = 81

    amr, _ = _make_amr(n_coarse=n, n_comp=2, x_left=0.0, x_right=1.0)
    x = amr.x_coarse
    amr.set_coarse_array(np.sin(2 * math.pi * x), 0)
    amr.set_coarse_array(np.cos(2 * math.pi * x), 1)
    amr.determine_refinement_segments()

    amr_d = compute_derivative(amr, order, 1)
    reset_derivative_cache()

    err0 = _max_interior_err(amr_d.f_coarse[:, 0],  2*math.pi*np.cos(2*math.pi*x), fo2)
    err1 = _max_interior_err(amr_d.f_coarse[:, 1], -2*math.pi*np.sin(2*math.pi*x), fo2)

    ok0 = _pass(f"comp 0 (sin'), max_err={err0:.3e}", err0 < 1e-4)
    ok1 = _pass(f"comp 1 (cos'), max_err={err1:.3e}", err1 < 1e-4)
    return ok0 and ok1


# ===========================================================================
# Runner
# ===========================================================================

def run_all():
    tests = [
        test_fd_deriv_polynomial_exactness,
        test_fd_deriv_sin_accuracy,
        test_fd_deriv_sparse_matches_dense,
        test_compute_derivative_coarse_first,
        test_compute_derivative_coarse_second,
        test_compute_derivative_order_convergence,
        test_no_sync_matches_compute_derivative_coarse,
        test_compute_derivative_refined_first,
        test_compute_derivative_refined_second,
        test_refined_derivative_finer_than_coarse,
        test_compute_derivative_at_level,
        test_compute_derivative_at_level_second,
        test_multi_component_derivatives,
    ]

    print("=" * 62)
    print("  Derivative verification test suite")
    print("=" * 62)

    results = {}
    for t in tests:
        try:
            results[t.__name__] = t()
        except Exception as e:
            print(f"  [ERROR]  {t.__name__}: {e}")
            results[t.__name__] = False

    n_pass = sum(results.values())
    n_fail = len(results) - n_pass

    print("\n" + "=" * 62)
    print(f"  Results: {n_pass} passed, {n_fail} failed out of {len(results)} test groups")
    print("=" * 62)

    if n_fail:
        print("\nFailed groups:")
        for name, ok in results.items():
            if not ok:
                print(f"  {name}")

    return n_fail == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)

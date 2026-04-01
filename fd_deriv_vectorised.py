"""
fd_deriv_vectorised.py
Demonstrates vectorised replacements for the loop-based derivative methods
in fd_deriv.py.

Each method below is a drop-in replacement.  The key idea is the same in
both cases:

    INTERIOR (regular stencil)
    --------------------------
    The original code computes, for each interior point i:

        df[i] = sum_j  DN[fo2, j] * f[i + j - fo2]

    This is a 1-D convolution of f with the stencil row DN[fo2, :].
    NumPy can do the whole interior block in one call with np.convolve
    (or equivalently as a strided matrix-vector product).

    BOUNDARY (irregular stencil)
    ----------------------------
    Only fo2 rows at each end — a small (fo2 x op1) matrix times a short
    slice of f.  This is just a matrix-vector product: DN[:fo2, :] @ f[:op1].

The vectorised versions are typically 20-100x faster than the loop versions
for realistic grid sizes.
"""

import numpy as np
from fd_deriv import FDDeriv   # inherit everything else unchanged


class FDDerivFast(FDDeriv):
    """
    Drop-in replacement for FDDeriv with vectorised derivative methods.
    Constructor, normalize_derivatives, and form_deriv_matrices are
    inherited unchanged.
    """

    def compute_first_derivative(self, n_values: int, f_values) -> np.ndarray:
        f  = np.asarray(f_values, dtype=float)
        op1 = self.order + 1
        fo2 = self.order // 2
        df  = np.zeros(n_values)

        # ------------------------------------------------------------------
        # Boundary rows — small matrix-vector products
        # ------------------------------------------------------------------
        # Top:    df[:fo2]       = DN_1[:fo2, :]   @ f[:op1]
        # Bottom: df[-fo2:]      = DN_1[-fo2:, :]  @ f[-op1:]
        #
        # BEFORE (two nested loops):
        #   for i in range(fo2):
        #       for j in range(op1):
        #           df[i] += DN_1[i, j] * f[j]
        #
        # AFTER (one line):
        df[:fo2]  = self.DN_1[:fo2,  :] @ f[:op1]
        df[-fo2:] = self.DN_1[-fo2:, :] @ f[-op1:]

        # ------------------------------------------------------------------
        # Interior rows — convolution with the centred stencil row
        # ------------------------------------------------------------------
        # Every interior point uses the same stencil row DN_1[fo2, :].
        # The operation is:
        #   df[i] = sum_j  DN_1[fo2, j] * f[i + j - fo2]   for i in [fo2, n-fo2)
        #
        # That is exactly np.convolve(f, stencil[::-1], mode='full')[fo2:n-fo2+fo2]
        # but the cleanest NumPy expression uses stride tricks via
        # as_strided to build the sliding window matrix, then a single dot.
        #
        # BEFORE (two nested loops):
        #   for j in range(op1):
        #       offset = j - fo2
        #       for i in range(fo2, n_values - fo2):
        #           df[i] += DN_1[fo2, j] * f[i + offset]
        #
        # AFTER: build a (n_interior x op1) view of f with stride tricks,
        # then multiply by the stencil row in one dot product.

        n_int = n_values - 2 * fo2          # number of interior points

        # sliding_window[i, j] = f[i + j]  where i runs over interior points
        # and j runs over the op1 stencil positions (0-indexed, so j=0
        # corresponds to the left edge of the stencil at f[i]).
        from numpy.lib.stride_tricks import as_strided
        itemsize = f.strides[0]
        sliding_window = as_strided(
            f,
            shape=(n_int, op1),
            strides=(itemsize, itemsize),
        )
        # stencil row (already centred: position fo2 corresponds to f[i])
        stencil = self.DN_1[fo2, :]         # shape (op1,)
        df[fo2:-fo2] = sliding_window @ stencil

        return df

    def compute_second_derivative(self, n_values: int, f_values) -> np.ndarray:
        f   = np.asarray(f_values, dtype=float)
        op1 = self.order + 1
        fo2 = self.order // 2
        df  = np.zeros(n_values)

        # ------------------------------------------------------------------
        # Boundary rows — matrix-vector products (identical pattern to above)
        # ------------------------------------------------------------------
        df[:fo2]  = self.DN_2[:fo2,  :] @ f[:op1]
        df[-fo2:] = self.DN_2[-fo2:, :] @ f[-op1:]

        # ------------------------------------------------------------------
        # Interior rows — sliding window dot product
        # ------------------------------------------------------------------
        n_int = n_values - 2 * fo2

        from numpy.lib.stride_tricks import as_strided
        itemsize = f.strides[0]
        sliding_window = as_strided(
            f,
            shape=(n_int, op1),
            strides=(itemsize, itemsize),
        )
        stencil = self.DN_2[fo2, :]
        df[fo2:-fo2] = sliding_window @ stencil

        return df


# ===========================================================================
# Benchmark and correctness check
# ===========================================================================

if __name__ == '__main__':
    import math, timeit

    order = 4
    k     = 2 * math.pi

    # Build both objects
    slow = FDDeriv(order)
    fast = FDDerivFast(order)

    for n in [51, 201, 801, 3201]:
        x  = np.linspace(0.0, 1.0, n)
        dx = x[1] - x[0]
        f  = np.sin(k * x)
        df_exact  = k * np.cos(k * x)
        d2f_exact = -k**2 * np.sin(k * x)

        slow.normalize_derivatives(dx)
        fast.normalize_derivatives(dx)

        # ---- correctness ----
        df_slow  = slow.compute_first_derivative(n, f)
        df_fast  = fast.compute_first_derivative(n, f)
        d2_slow  = slow.compute_second_derivative(n, f)
        d2_fast  = fast.compute_second_derivative(n, f)

        assert np.allclose(df_slow,  df_fast,  atol=1e-10), "first deriv mismatch"
        assert np.allclose(d2_slow,  d2_fast,  atol=1e-10), "second deriv mismatch"

        fo2 = order // 2
        err1 = np.max(np.abs(df_fast[fo2:-fo2]  - df_exact[fo2:-fo2]))
        err2 = np.max(np.abs(d2_fast[fo2:-fo2]  - d2f_exact[fo2:-fo2]))

        # ---- timing ----
        reps = max(10, 2000 // n)
        t_slow_1 = timeit.timeit(lambda: slow.compute_first_derivative(n, f),  number=reps) / reps * 1e6
        t_fast_1 = timeit.timeit(lambda: fast.compute_first_derivative(n, f),  number=reps) / reps * 1e6
        t_slow_2 = timeit.timeit(lambda: slow.compute_second_derivative(n, f), number=reps) / reps * 1e6
        t_fast_2 = timeit.timeit(lambda: fast.compute_second_derivative(n, f), number=reps) / reps * 1e6

        print(f'\nn = {n:5d}   err1={err1:.1e}   err2={err2:.1e}')
        print(f'  first  deriv:  slow={t_slow_1:7.1f} µs   fast={t_fast_1:6.1f} µs   speedup={t_slow_1/t_fast_1:5.1f}x')
        print(f'  second deriv:  slow={t_slow_2:7.1f} µs   fast={t_fast_2:6.1f} µs   speedup={t_slow_2/t_fast_2:5.1f}x')

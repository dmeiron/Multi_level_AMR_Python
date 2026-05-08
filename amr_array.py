"""
amr_array.py
Adaptive Mesh Refinement (AMR) array class, converted from MATLAB.

Key mapping notes:
  - MATLAB cell arrays  → Python dicts (keyed by (i, lev), 0-indexed)
  - MATLAB 1-based indexing → Python 0-based indexing throughout
  - MATLAB classdef / methods → Python class with regular methods
  - Operator overloading: +, -, *, / preserved via __add__, __sub__, etc.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Problem-definition dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProbDef:
    """
    Problem definition passed to AMRArray.

    Attributes
    ----------
    n_coarse    : number of coarse grid points
    n_comp      : number of solution components
    max_no_levs : maximum number of refinement levels (including coarse)
    ref_fac     : refinement factor between levels
    x_left      : left boundary of the domain
    x_right     : right boundary of the domain
    """
    n_coarse:    int
    n_comp:      int
    max_no_levs: int
    ref_fac:     int
    x_left:      float
    x_right:     float


# ---------------------------------------------------------------------------
# AMRArray
# ---------------------------------------------------------------------------

class AMRArray:
    """
    Multi-level Adaptive Mesh Refinement array.

    Parameters
    ----------
    prob_def : ProbDef
        Problem definition (see ProbDef dataclass).
    """

    # Declare all instance attributes so Pylance can see their types without
    # having to infer them through prob_def's fields.
    n_coarse:         int
    n_comp:           int
    max_no_levs:      int
    ref_fac:          int
    x_left:           float
    x_right:          float
    ref_levs_so_far:  int
    n_ref:            NDArray[np.int_]
    dx:               NDArray[np.float64]
    is_ref:           NDArray[np.bool_]
    n_ref_seg:        NDArray[np.int_]
    beg_ref_seg:      NDArray[np.int_]
    end_ref_seg:      NDArray[np.int_]
    tot_ref_lev:      NDArray[np.int_]
    coarse_lev_depth: NDArray[np.int_]
    x_coarse:         NDArray[np.float64]
    f_coarse:         NDArray[np.float64]
    f_arr:            dict[tuple[int, int], NDArray[np.float64]]
    x_coord:          dict[tuple[int, int], NDArray[np.float64]]

    def __init__(self, prob_def: ProbDef) -> None:
        # ---- basic scalars -----------------------------------------------
        self.n_coarse    = prob_def.n_coarse
        self.n_comp      = prob_def.n_comp
        self.max_no_levs = prob_def.max_no_levs
        self.ref_fac     = prob_def.ref_fac
        self.x_left      = prob_def.x_left
        self.x_right     = prob_def.x_right

        self.ref_levs_so_far: int = 0

        # ---- number of fine points inside one coarse interval per level ---
        # n_ref[lev] = ref_fac^lev  (0-indexed levels)
        self.n_ref = np.array(
            [self.ref_fac ** lev for lev in range(self.max_no_levs)],
            dtype=int,
        )

        # ---- mesh spacing per level ---------------------------------------
        coarse_dx: float = (self.x_right - self.x_left) / (self.n_coarse - 1)
        self.dx = np.array(
            [coarse_dx / self.n_ref[lev] for lev in range(self.max_no_levs)],
            dtype=float,
        )

        # ---- refinement flag: shape (n_coarse, max_no_levs) --------------
        self.is_ref = np.zeros((self.n_coarse, self.max_no_levs), dtype=bool)

        # ---- segment tracking --------------------------------------------
        self.n_ref_seg   = np.zeros(self.max_no_levs, dtype=int)
        self.beg_ref_seg = np.zeros((self.n_coarse, self.max_no_levs), dtype=int)
        self.end_ref_seg = np.zeros((self.n_coarse, self.max_no_levs), dtype=int)

        # ---- total refined intervals per level ---------------------------
        self.tot_ref_lev = np.zeros(self.max_no_levs, dtype=int)
        self.tot_ref_lev[0] = self.n_coarse          # level 0 = coarse grid

        # ---- depth (highest level touched) at each coarse point ----------
        self.coarse_lev_depth = np.zeros(self.n_coarse, dtype=int)

        # ---- coarse x coordinates ----------------------------------------
        self.x_coarse = np.linspace(self.x_left, self.x_right, self.n_coarse)

        # ---- coarse function values: shape (n_coarse, n_comp) ------------
        self.f_coarse = np.zeros((self.n_coarse, self.n_comp))

        # ---- cell arrays as Python dicts keyed by (i, lev) ---------------
        self.f_arr:   dict[tuple[int, int], NDArray[np.float64]] = {}
        self.x_coord: dict[tuple[int, int], NDArray[np.float64]] = {}

        # ---- initialise level-0 x_coord segments -------------------------
        n_r: int = int(self.n_ref[0])   # = 1
        for i in range(self.n_coarse - 1):
            self.x_coord[(i, 0)] = self.x_coarse[i : i + n_r + 1].copy()

    # -----------------------------------------------------------------------
    # Coarse-array helpers
    # -----------------------------------------------------------------------

    def set_coarse_array(self, values: NDArray[np.float64], comp: int) -> None:
        """
        Set the entire coarse array for component `comp` (0-indexed).

        Parameters
        ----------
        values : array-like, length n_coarse
        comp   : int, 0-indexed component index
        """
        self.f_coarse[:, comp] = values
        self.set_contiguous_refinement_array(
            level=0, comp=comp,
            index_start=0, index_end=self.n_coarse - 2,
            values=values,
        )

    def get_coarse_array(self, comp: int) -> NDArray[np.float64]:
        """Return the coarse array for component `comp` (0-indexed)."""
        return self.f_coarse[:, comp].copy()

    # -----------------------------------------------------------------------
    # Refinement array access
    # -----------------------------------------------------------------------

    def set_refinement_at(
        self,
        index: int,
        level: int,
        values: NDArray[np.float64],
    ) -> AMRArray:
        """
        Store refined values at coarse point `index`, refinement level `level`.

        Parameters
        ----------
        index  : int – coarse grid index (0-indexed)
        level  : int – refinement level (0-indexed; 0 = coarse)
        values : ndarray, shape (n_ref[level]+1, n_comp)
        """
        n_r: int = int(self.n_ref[level])
        self.f_arr[(index, level)] = values[:n_r + 1, :].copy()

        if level > 0:
            self.is_ref[index, level - 1] = True

        self.tot_ref_lev[level] += 1
        self.coarse_lev_depth[index] = level

        if level > self.ref_levs_so_far:
            self.ref_levs_so_far = level

        return self

    def get_refinement_at(
        self, index: int, level: int
    ) -> NDArray[np.float64] | None:
        """
        Return a *copy* of the refined-value array at (index, level).
        Returns None if no data has been stored there.
        """
        arr = self.f_arr.get((index, level))
        return arr.copy() if arr is not None else None

    # -----------------------------------------------------------------------
    # Contiguous segment helpers
    # -----------------------------------------------------------------------

    def get_contiguous_refinement_array(
        self,
        level: int,
        comp: int,
        index_start: int,
        index_end: int,
    ) -> NDArray[np.float64]:
        """
        Return a 1-D array of function values for component `comp` spanning
        coarse indices [index_start, index_end] at the given level.
        """
        n_r: int = int(self.n_ref[level])
        n_values: int = (index_end - index_start + 1) * n_r + 1
        values: NDArray[np.float64] = np.zeros(n_values)

        idx = 0
        for i in range(index_start, index_end + 1):
            arr = self.f_arr.get((i, level))
            if arr is not None:
                for j in range(n_r):
                    values[idx] = arr[j, comp]
                    idx += 1
        arr = self.f_arr.get((index_end, level))
        if arr is not None:
            values[idx] = arr[n_r, comp]

        return values

    def set_contiguous_refinement_array(
        self,
        level: int,
        comp: int,
        index_start: int,
        index_end: int,
        values: NDArray[np.float64],
    ) -> None:
        """
        Store `values` into the contiguous refinement array spanning
        [index_start, index_end] at `level` for component `comp`.
        """
        n_r: int = int(self.n_ref[level])
        idx = 0
        for i in range(index_start, index_end + 1):
            seg: NDArray[np.float64] = np.zeros((n_r + 1, self.n_comp))
            existing = self.f_arr.get((i, level))
            if existing is not None:
                seg[:] = existing

            for j in range(n_r + 1):
                seg[j, comp] = values[idx]
                idx += 1
            idx -= 1   # overlap: last point of this segment = first of next
            self.f_arr[(i, level)] = seg

    def get_contiguous_refinement_x_coord(
        self,
        level: int,
        index_start: int,
        index_end: int,
    ) -> NDArray[np.float64]:
        """
        Return a 1-D array of x coordinates spanning [index_start, index_end]
        at `level`.
        """
        n_r: int = int(self.n_ref[level])
        n_values: int = (index_end - index_start + 1) * n_r + 1
        values: NDArray[np.float64] = np.zeros(n_values)

        idx = 0
        for i in range(index_start, index_end + 1):
            xc = self.x_coord.get((i, level))
            if xc is not None:
                for j in range(n_r):
                    values[idx] = xc[j]
                    idx += 1
        xc = self.x_coord.get((index_end, level))
        if xc is not None:
            values[idx] = xc[n_r]

        return values

    # -----------------------------------------------------------------------
    # Refinement segment bookkeeping
    # -----------------------------------------------------------------------

    def determine_refinement_segments(self) -> None:
        """
        Scan is_ref and populate n_ref_seg, beg_ref_seg, end_ref_seg.
        """
        for i_lev in range(self.ref_levs_so_far):
            seg_count = 0
            in_seg = False
            for i in range(self.n_coarse):
                if self.is_ref[i, i_lev]:
                    if not in_seg:
                        self.beg_ref_seg[seg_count, i_lev] = i
                        in_seg = True
                else:
                    if in_seg:
                        self.end_ref_seg[seg_count, i_lev] = i - 1
                        seg_count += 1
                        in_seg = False
            if in_seg:
                self.end_ref_seg[seg_count, i_lev] = self.n_coarse - 1
                seg_count += 1
            self.n_ref_seg[i_lev] = seg_count

    # -----------------------------------------------------------------------
    # Deletion
    # -----------------------------------------------------------------------

    def delete_refinement_array(self, level: int, array_index: int) -> None:
        """Remove the refinement data at (array_index, level)."""
        if not (0 <= array_index < self.n_coarse):
            raise IndexError("array_index out of bounds")

        flag_set = (
            self.is_ref[array_index, level - 1] if level > 0
            else (array_index, level) in self.f_arr
        )

        if flag_set:
            self.f_arr.pop((array_index, level), None)
            self.x_coord.pop((array_index, level), None)
            if level > 0:
                self.is_ref[array_index, level - 1] = False
            self.tot_ref_lev[level] -= 1
            self.coarse_lev_depth[array_index] -= 1
        else:
            raise RuntimeError(
                f"Refinement at index {array_index}, level {level} is already empty."
            )

    def delete_bottom_level(self) -> None:
        """Remove the finest refinement level from all coarse points."""
        bottom_level: int = self.ref_levs_so_far
        for i in range(self.n_coarse):
            if self.is_ref[i, bottom_level - 1]:
                self.f_arr.pop((i, bottom_level), None)
                self.x_coord.pop((i, bottom_level), None)
                self.is_ref[i, bottom_level - 1] = False
                self.tot_ref_lev[bottom_level] -= 1
                self.coarse_lev_depth[i] -= 1

        self.ref_levs_so_far -= 1

    # -----------------------------------------------------------------------
    # Collapse to a flat array for plotting
    # -----------------------------------------------------------------------

    def collapse_array(
        self, comp: int, start: int, finish: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Build flat (x, f) arrays spanning coarse indices [start, finish]
        using the highest available refinement at each point.

        Returns
        -------
        x_arr   : 1-D ndarray of x coordinates
        col_arr : 1-D ndarray of function values for component `comp`
        """
        x_list:   list[NDArray[np.float64]] = []
        col_list: list[NDArray[np.float64]] = []

        for i in range(start, finish + 1):
            ml:  int = int(self.coarse_lev_depth[i])
            n_r: int = int(self.n_ref[ml])

            values   = self.f_arr.get((i, ml))
            x_values = self.x_coord.get((i, ml))

            if values is None or x_values is None:
                continue

            x_list.append(x_values[:n_r])
            col_list.append(values[:n_r, comp])

        if x_list:
            return np.concatenate(x_list), np.concatenate(col_list)
        return np.array([]), np.array([])

    # -----------------------------------------------------------------------
    # Apply a function to every level
    # -----------------------------------------------------------------------

    def apply_function(
        self, func: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    ) -> AMRArray:
        """
        Apply a callable `func(x) -> f` to every level of the AMR structure.
        Returns a new AMRArray with the evaluated values.
        """
        obj_out = copy.deepcopy(self)

        raw = func(self.x_coarse)
        if np.ndim(raw) == 1:
            for c in range(self.n_comp):
                obj_out.f_coarse[:, c] = raw
        else:
            obj_out.f_coarse = raw

        for i in range(self.n_coarse - 1):
            for lev in range(self.coarse_lev_depth[i] + 1):
                xc = self.x_coord.get((i, lev))
                if xc is None:
                    continue
                raw = func(xc)
                arr: NDArray[np.float64] = np.zeros((len(xc), self.n_comp))
                if np.ndim(raw) == 1:
                    for c in range(self.n_comp):
                        arr[:, c] = raw
                else:
                    arr = raw
                obj_out.f_arr[(i, lev)] = arr
                obj_out.x_coord[(i, lev)] = xc.copy()

        return obj_out

    # -----------------------------------------------------------------------
    # Fine-to-coarse transfer
    # -----------------------------------------------------------------------

    def fine_to_coarse(self) -> AMRArray:
        """
        Transfer finest values down to coarser levels (in-place).
        Returns self for chaining.
        """
        nc: int = self.n_comp
        rf: int = self.ref_fac

        for i_coarse in range(self.n_coarse - 1):
            for i_lev in range(self.coarse_lev_depth[i_coarse], 0, -1):
                fine   = self.f_arr.get((i_coarse, i_lev))
                coarse = self.f_arr.get((i_coarse, i_lev - 1))
                if fine is None or coarse is None:
                    continue

                n_r_coarse: int = int(self.n_ref[i_lev - 1])
                for i in range(n_r_coarse + 1):
                    coarse[i, :nc] = fine[i * rf, :nc]
                self.f_arr[(i_coarse, i_lev - 1)] = coarse

                right_coarse = self.f_arr.get((i_coarse + 1, i_lev - 1))
                if right_coarse is not None:
                    right_coarse[0, :nc] = fine[int(self.n_ref[i_lev]), :nc]
                    self.f_arr[(i_coarse + 1, i_lev - 1)] = right_coarse

                if i_coarse > 0:
                    left_coarse = self.f_arr.get((i_coarse - 1, i_lev - 1))
                    if left_coarse is not None:
                        left_coarse[int(self.n_ref[i_lev - 1]), :nc] = fine[0, :nc]
                        self.f_arr[(i_coarse - 1, i_lev - 1)] = left_coarse

            arr0 = self.f_arr.get((i_coarse, 0))
            if arr0 is not None:
                self.f_coarse[i_coarse, :nc] = arr0[0, :nc]

        arr_last = self.f_arr.get((self.n_coarse - 2, 0))
        if arr_last is not None:
            self.f_coarse[self.n_coarse - 1, :nc] = arr_last[1, :nc]

        return self

    # -----------------------------------------------------------------------
    # Operator overloading
    # -----------------------------------------------------------------------

    def _apply_op(
        self,
        other: AMRArray | float | int,
        op: Callable[[NDArray[np.float64], NDArray[np.float64] | float], NDArray[np.float64]],
    ) -> AMRArray:
        """Helper: apply a binary operator element-wise to all levels."""
        result = copy.deepcopy(self)

        if isinstance(other, (int, float)):
            scalar = float(other)  # type: ignore[arg-type]
            result.f_coarse = op(self.f_coarse, scalar)
            for i_pt in range(self.n_coarse):
                for i_lev in range(int(self.coarse_lev_depth[i_pt]) + 1):
                    arr = self.f_arr.get((i_pt, i_lev))
                    if arr is not None:
                        result.f_arr[(i_pt, i_lev)] = op(arr, scalar)

        elif hasattr(other, "f_coarse"):
            result.f_coarse = op(self.f_coarse, other.f_coarse)  # type: ignore[arg-type]
            for i_pt in range(self.n_coarse):
                for i_lev in range(int(self.coarse_lev_depth[i_pt]) + 1):
                    a = self.f_arr.get((i_pt, i_lev))
                    b = other.f_arr.get((i_pt, i_lev))  # type: ignore[union-attr]
                    if a is not None and b is not None:
                        result.f_arr[(i_pt, i_lev)] = op(a, b)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return result

    def __add__(self, other: AMRArray | float | int) -> AMRArray:
        return self._apply_op(other, lambda a, b: a + b)

    def __radd__(self, other: AMRArray | float | int) -> AMRArray:
        return self._apply_op(other, lambda a, b: b + a)

    def __sub__(self, other: AMRArray | float | int) -> AMRArray:
        return self._apply_op(other, lambda a, b: a - b)

    def __rsub__(self, other: AMRArray | float | int) -> AMRArray:
        return self._apply_op(other, lambda a, b: b - a)

    def __mul__(self, other: AMRArray | float | int) -> AMRArray:
        return self._apply_op(other, lambda a, b: a * b)

    def __rmul__(self, other: AMRArray | float | int) -> AMRArray:
        return self._apply_op(other, lambda a, b: b * a)

    def __truediv__(self, other: AMRArray | float | int) -> AMRArray:
        return self._apply_op(other, lambda a, b: a / b)

    def __neg__(self) -> AMRArray:
        result = copy.deepcopy(self)
        result.f_coarse = -self.f_coarse
        for key, arr in self.f_arr.items():
            result.f_arr[key] = -arr
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable summary of the AMR structure."""
        print(f"AMRArray: n_coarse={self.n_coarse}, n_comp={self.n_comp}, "
              f"ref_fac={self.ref_fac}, ref_levs_so_far={self.ref_levs_so_far}")
        print(f"  Domain: [{self.x_left:.4g}, {self.x_right:.4g}]  "
              f"dx[0]={self.dx[0]:.4g}")
        print(f"  Coarse f_coarse range: "
              f"[{self.f_coarse.min():.4g}, {self.f_coarse.max():.4g}]")

        if self.ref_levs_so_far == 0:
            print("  No refined levels.")
            return

        for i_lev in range(1, self.ref_levs_so_far + 1):
            n_seg = int(self.n_ref_seg[i_lev - 1])
            print(f"  Level {i_lev}: {n_seg} segment(s),  dx={self.dx[i_lev]:.4g}")
            for i_seg in range(n_seg):
                beg = int(self.beg_ref_seg[i_seg, i_lev - 1])
                end = int(self.end_ref_seg[i_seg, i_lev - 1])
                n_pts = 0
                vals_min, vals_max = float('inf'), float('-inf')
                for i_pt in range(beg, end + 1):
                    arr = self.f_arr.get((i_pt, i_lev))
                    if arr is not None:
                        n_pts += len(arr)
                        vals_min = min(vals_min, float(arr.min()))
                        vals_max = max(vals_max, float(arr.max()))
                        print(f"    seg {i_seg}: coarse pts [{beg}, {end}]  "
                              f"fine pts={n_pts}  "
                              f"f range=[{vals_min:.4g}, {vals_max:.4g}]")

    

    # -----------------------------------------------------------------------
    # Convenience / debug
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AMRArray(n_coarse={self.n_coarse}, n_comp={self.n_comp}, "
            f"max_no_levs={self.max_no_levs}, ref_fac={self.ref_fac}, "
            f"ref_levs_so_far={self.ref_levs_so_far})"
        )
    

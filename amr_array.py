"""
amr_array.py
Adaptive Mesh Refinement (AMR) array class, converted from MATLAB.

Key mapping notes:
  - MATLAB cell arrays  → Python dicts (keyed by (i, lev), 0-indexed)
  - MATLAB 1-based indexing → Python 0-based indexing throughout
  - MATLAB classdef / methods → Python class with regular methods
  - Operator overloading: +, -, *, / preserved via __add__, __sub__, etc.
"""

import numpy as np
import copy


class AMRArray:
    """
    Multi-level Adaptive Mesh Refinement array.

    Parameters
    ----------
    prob_def : object or dict-like
        Must expose the attributes:
            n_coarse    – number of coarse grid points
            n_comp      – number of solution components
            max_no_levs – maximum number of refinement levels
            ref_fac     – refinement factor between levels
            x_left      – left boundary of the domain
            x_right     – right boundary of the domain
    """

    def __init__(self, prob_def):
        # ---- basic scalars -----------------------------------------------
        self.n_coarse    = prob_def.n_coarse
        self.n_comp      = prob_def.n_comp
        self.max_no_levs = prob_def.max_no_levs
        self.ref_fac     = prob_def.ref_fac
        self.x_left      = prob_def.x_left
        self.x_right     = prob_def.x_right

        self.ref_levs_so_far = 0

        # ---- number of fine points inside one coarse interval per level ---
        # n_ref[lev] = ref_fac^lev  (0-indexed levels)
        self.n_ref = np.array(
            [self.ref_fac ** lev for lev in range(self.max_no_levs)],
            dtype=int
        )

        # ---- mesh spacing per level ---------------------------------------
        coarse_dx = (self.x_right - self.x_left) / (self.n_coarse - 1)
        self.dx = np.array(
            [coarse_dx / self.n_ref[lev] for lev in range(self.max_no_levs)]
        )

        # ---- refinement flag: shape (n_coarse, max_no_levs), 0-indexed ---
        self.is_ref = np.zeros((self.n_coarse, self.max_no_levs), dtype=bool)

        # ---- segment tracking --------------------------------------------
        self.n_ref_seg  = np.zeros(self.max_no_levs, dtype=int)
        self.beg_ref_seg = np.zeros((self.n_coarse, self.max_no_levs), dtype=int)
        self.end_ref_seg = np.zeros((self.n_coarse, self.max_no_levs), dtype=int)

        # ---- total refined intervals per level ---------------------------
        self.tot_ref_lev = np.zeros(self.max_no_levs, dtype=int)
        self.tot_ref_lev[0] = self.n_coarse          # level 0 = coarse grid

        # ---- depth (highest level touched) at each coarse point ----------
        # 0-indexed: 0 means only the coarse level exists
        self.coarse_lev_depth = np.zeros(self.n_coarse, dtype=int)

        # ---- coarse x coordinates ----------------------------------------
        self.x_coarse = np.linspace(self.x_left, self.x_right, self.n_coarse)

        # ---- coarse function values: shape (n_coarse, n_comp) ------------
        self.f_coarse = np.zeros((self.n_coarse, self.n_comp))

        # ---- cell arrays as Python dicts keyed by (i, lev) ---------------
        # f_arr   : stores function values arrays at each (coarse_pt, level)
        # x_coord : stores x-coordinate arrays at each (coarse_pt, level)
        self.f_arr   = {}   # (i, lev) -> ndarray shape (n_ref[lev]+1, n_comp)
        self.x_coord = {}   # (i, lev) -> ndarray shape (n_ref[lev]+1,)

        # ---- initialise level-0 x_coord segments -------------------------
        # Each coarse interval i covers x_coarse[i] .. x_coarse[i+1]
        # At level 0, n_ref[0] == 1, so each segment has 2 points.
        n_r = self.n_ref[0]   # = 1
        for i in range(self.n_coarse - 1):
            self.x_coord[(i, 0)] = self.x_coarse[i : i + n_r + 1].copy()

    # -----------------------------------------------------------------------
    # Coarse-array helpers
    # -----------------------------------------------------------------------

    def set_coarse_array(self, values, comp):
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
            values=values
        )

    def get_coarse_array(self, comp):
        """Return the coarse array for component `comp` (0-indexed)."""
        return self.f_coarse[:, comp].copy()

    # -----------------------------------------------------------------------
    # Refinement array access
    # -----------------------------------------------------------------------

    def set_refinement_at(self, index, level, values):
        """
        Store refined values at coarse point `index`, refinement level `level`
        (both 0-indexed).

        Parameters
        ----------
        index  : int – coarse grid index (0-indexed)
        level  : int – refinement level (0-indexed; 0 = coarse)
        values : ndarray, shape (n_ref[level]+1, n_comp)
        """
        n_r = self.n_ref[level]
        self.f_arr[(index, level)] = values[:n_r + 1, :].copy()

        # The *previous* level is now flagged as refined at this index
        if level > 0:
            self.is_ref[index, level - 1] = True

        self.tot_ref_lev[level] += 1
        self.coarse_lev_depth[index] = level

        if level > self.ref_levs_so_far:
            self.ref_levs_so_far = level

        return self

    def get_refinement_at(self, index, level):
        """
        Return the refined-value array at (index, level).

        Returns None if no data has been stored there.
        """
        return self.f_arr.get((index, level), None)

    # -----------------------------------------------------------------------
    # Contiguous segment helpers
    # -----------------------------------------------------------------------

    def get_contiguous_refinement_array(self, level, comp, index_start, index_end):
        """
        Return a 1-D array of function values for component `comp` spanning
        coarse indices [index_start, index_end] at the given level.
        All 0-indexed.
        """
        n_r = self.n_ref[level]
        n_values = (index_end - index_start + 1) * n_r + 1
        values = np.zeros(n_values)

        idx = 0
        for i in range(index_start, index_end + 1):
            arr = self.f_arr.get((i, level))
            for j in range(n_r):
                values[idx] = arr[j, comp]
                idx += 1
        # last point
        arr = self.f_arr.get((index_end, level))
        values[idx] = arr[n_r, comp]

        return values

    def set_contiguous_refinement_array(self, level, comp, index_start, index_end, values):
        """
        Store `values` into the contiguous refinement array spanning
        [index_start, index_end] at `level` for component `comp`.
        The end value of each segment is shared with the start of the next.
        All 0-indexed.
        """
        n_r = self.n_ref[level]
        idx = 0
        for i in range(index_start, index_end + 1):
            seg = np.zeros((n_r + 1, self.n_comp))
            # preserve existing data for other components if entry exists
            existing = self.f_arr.get((i, level))
            if existing is not None:
                seg[:] = existing

            for j in range(n_r + 1):
                seg[j, comp] = values[idx]
                idx += 1
            idx -= 1   # overlap: last point of this segment = first of next
            self.f_arr[(i, level)] = seg

    def get_contiguous_refinement_x_coord(self, level, index_start, index_end):
        """
        Return a 1-D array of x coordinates spanning [index_start, index_end]
        at `level`. All 0-indexed.
        """
        n_r = self.n_ref[level]
        n_values = (index_end - index_start + 1) * n_r + 1
        values = np.zeros(n_values)

        idx = 0
        for i in range(index_start, index_end + 1):
            xc = self.x_coord.get((i, level))
            for j in range(n_r):
                values[idx] = xc[j]
                idx += 1
        xc = self.x_coord.get((index_end, level))
        values[idx] = xc[n_r]

        return values

    # -----------------------------------------------------------------------
    # Refinement segment detection
    # -----------------------------------------------------------------------

    def determine_refinement_segments(self):
        """
        Scan the AMR structure to identify contiguous refined segments at each
        refinement level.  Results stored in n_ref_seg, beg_ref_seg, end_ref_seg.
        """
        npts = self.n_coarse

        for i_lev in range(self.ref_levs_so_far):
            i_ref_loc = 0
            ref_loc_beg = np.zeros(npts, dtype=int)
            ref_loc_end = np.zeros(npts, dtype=int)

            # First point (i=0)
            if self.is_ref[0, i_lev]:
                i_ref_loc = 1
                ref_loc_beg[i_ref_loc - 1] = 0
                if not self.is_ref[1, i_lev]:
                    ref_loc_end[i_ref_loc - 1] = 0

            for i in range(1, npts - 1):
                is_ref_im1 = self.is_ref[i - 1, i_lev]
                is_ref_i   = self.is_ref[i,     i_lev]
                is_ref_ip1 = self.is_ref[i + 1, i_lev]

                if is_ref_i and not is_ref_im1:   # start of refined region
                    ref_loc_beg[i_ref_loc] = i
                    i_ref_loc += 1

                if is_ref_i and not is_ref_ip1:   # end of refined region
                    ref_loc_end[i_ref_loc - 1] = i

            n_seg = i_ref_loc
            self.n_ref_seg[i_lev]            = n_seg
            self.beg_ref_seg[:n_seg, i_lev]  = ref_loc_beg[:n_seg]
            self.end_ref_seg[:n_seg, i_lev]  = ref_loc_end[:n_seg]

    # -----------------------------------------------------------------------
    # Deletion helpers
    # -----------------------------------------------------------------------

    def delete_refinement_array(self, level, array_index):
        """
        Remove the refined data at (array_index, level) and mark as unrefined.
        Both 0-indexed.
        """
        if not (0 <= array_index < self.n_coarse):
            raise IndexError("array_index out of bounds")

        if self.is_ref[array_index, level]:
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

    def delete_bottom_level(self):
        """Remove the finest refinement level from all coarse points."""
        bottom_level = self.ref_levs_so_far + 1
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

    def collapse_array(self, comp, start, finish):
        """
        Build flat (x, f) arrays spanning coarse indices [start, finish]
        using the highest available refinement at each point.
        All 0-indexed.

        Returns
        -------
        x_arr   : 1-D ndarray of x coordinates
        col_arr : 1-D ndarray of function values for component `comp`
        """
        x_list   = []
        col_list = []

        for i in range(start, finish + 1):
            ml  = self.coarse_lev_depth[i]
            n_r = self.n_ref[ml]

            values   = self.f_arr.get((i, ml))
            x_values = self.x_coord.get((i, ml))

            if values is None or x_values is None:
                continue

            x_list.append(x_values[:n_r])
            col_list.append(values[:n_r, comp])

        if x_list:
            return np.concatenate(x_list), np.concatenate(col_list)
        else:
            return np.array([]), np.array([])

    # -----------------------------------------------------------------------
    # Apply a function to every level
    # -----------------------------------------------------------------------

    def apply_function(self, func):
        """
        Apply a callable `func(x) -> f` to every level of the AMR structure.
        Returns a new AMRArray with the evaluated values.

        `func` should accept a 1-D ndarray and return a 1-D ndarray (or scalar).
        """
        obj_out = copy.deepcopy(self)

        raw = func(self.x_coarse)
        # Store as (n_coarse, n_comp); if func returns 1-D, broadcast
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
                arr = np.zeros((len(xc), self.n_comp))
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

    def fine_to_coarse(self):
        """
        Transfer finest values down to coarser levels (in-place).
        Returns self for chaining.
        """
        nc = self.n_comp
        rf = self.ref_fac

        for i_coarse in range(self.n_coarse - 1):
            for i_lev in range(self.coarse_lev_depth[i_coarse], 0, -1):
                fine   = self.f_arr.get((i_coarse, i_lev))
                coarse = self.f_arr.get((i_coarse, i_lev - 1))
                if fine is None or coarse is None:
                    continue

                n_r_coarse = self.n_ref[i_lev - 1]
                for i in range(n_r_coarse + 1):
                    coarse[i, :nc] = fine[(i) * rf, :nc]
                self.f_arr[(i_coarse, i_lev - 1)] = coarse

                # propagate endpoint to next coarse interval
                right_coarse = self.f_arr.get((i_coarse + 1, i_lev - 1))
                if right_coarse is not None:
                    right_coarse[0, :nc] = fine[self.n_ref[i_lev], :nc]
                    self.f_arr[(i_coarse + 1, i_lev - 1)] = right_coarse

                # propagate startpoint to previous coarse interval
                if i_coarse > 0:
                    left_coarse = self.f_arr.get((i_coarse - 1, i_lev - 1))
                    if left_coarse is not None:
                        left_coarse[self.n_ref[i_lev - 1], :nc] = fine[0, :nc]
                        self.f_arr[(i_coarse - 1, i_lev - 1)] = left_coarse

            # update coarse values from level-0 array
            arr0 = self.f_arr.get((i_coarse, 0))
            if arr0 is not None:
                self.f_coarse[i_coarse, :nc] = arr0[0, :nc]

        # last coarse point
        arr_last = self.f_arr.get((self.n_coarse - 2, 0))
        if arr_last is not None:
            self.f_coarse[self.n_coarse - 1, :nc] = arr_last[1, :nc]

        return self

    # -----------------------------------------------------------------------
    # Operator overloading
    # -----------------------------------------------------------------------

    def _apply_op(self, other, op):
        """Helper: apply a binary operator element-wise to all levels."""
        result = copy.deepcopy(self)

        if np.isscalar(other):
            result.f_coarse = op(self.f_coarse, other)
            for i_pt in range(self.n_coarse):
                for i_lev in range(self.coarse_lev_depth[i_pt] + 1):
                    arr = self.f_arr.get((i_pt, i_lev))
                    if arr is not None:
                        result.f_arr[(i_pt, i_lev)] = op(arr, other)

        elif isinstance(other, AMRArray):
            result.f_coarse = op(self.f_coarse, other.f_coarse)
            for i_pt in range(self.n_coarse):
                for i_lev in range(self.coarse_lev_depth[i_pt] + 1):
                    a = self.f_arr.get((i_pt, i_lev))
                    b = other.f_arr.get((i_pt, i_lev))
                    if a is not None and b is not None:
                        result.f_arr[(i_pt, i_lev)] = op(a, b)
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

        return result

    def __add__(self, other):
        return self._apply_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._apply_op(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._apply_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._apply_op(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._apply_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._apply_op(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._apply_op(other, lambda a, b: a / b)

    def __neg__(self):
        result = copy.deepcopy(self)
        result.f_coarse = -self.f_coarse
        for key, arr in self.f_arr.items():
            result.f_arr[key] = -arr
        return result

    # -----------------------------------------------------------------------
    # Convenience / debug
    # -----------------------------------------------------------------------

    def __repr__(self):
        return (
            f"AMRArray(n_coarse={self.n_coarse}, n_comp={self.n_comp}, "
            f"max_no_levs={self.max_no_levs}, ref_fac={self.ref_fac}, "
            f"ref_levs_so_far={self.ref_levs_so_far})"
        )

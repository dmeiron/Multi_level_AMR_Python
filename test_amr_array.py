"""
test_amr_array.py
Regression tests for AMRArray (amr_array.py).

Run with:  pytest test_amr_array.py -v
"""

import copy
import numpy as np
import pytest
import sys

# Allow importing from the uploads directory
sys.path.insert(0, "/mnt/user-data/outputs")
from amr_array import AMRArray, ProbDef


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------
def make_amr(**kwargs: int | float) -> AMRArray:
    """Convenience factory."""
    defaults: dict[str, int | float] = dict(
        n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2, x_left=0.0, x_right=1.0
    )
    defaults.update(kwargs)
    return AMRArray(ProbDef(
        n_coarse    = int(defaults['n_coarse']),
        n_comp      = int(defaults['n_comp']),
        max_no_levs = int(defaults['max_no_levs']),
        ref_fac     = int(defaults['ref_fac']),
        x_left      = float(defaults['x_left']),
        x_right     = float(defaults['x_right']),
    ))


def _add_level1_refinement(amr, index=1):
    """
    Store a simple level-1 refinement at coarse point `index`.
    n_ref[1] = ref_fac = 2, so we need a (3, n_comp) array.
    """
    n_r  = amr.n_ref[1]  # 2
    vals = np.arange((n_r + 1) * amr.n_comp, dtype=float).reshape(n_r + 1, amr.n_comp)
    x_fine = np.linspace(amr.x_coarse[index], amr.x_coarse[index + 1], n_r + 1)
    amr.x_coord[(index, 1)] = x_fine
    amr.set_refinement_at(index, 1, vals)
    return vals


# ===========================================================================
# 1. Initialisation
# ===========================================================================

class TestInit:
    def test_scalar_attributes(self):
        amr = make_amr(n_coarse=5, n_comp=2, max_no_levs=3, ref_fac=2)
        assert amr.n_coarse    == 5
        assert amr.n_comp      == 2
        assert amr.max_no_levs == 3
        assert amr.ref_fac     == 2
        assert amr.x_left      == 0.0
        assert amr.x_right     == 1.0
        assert amr.ref_levs_so_far == 0

    def test_n_ref_values(self):
        amr = make_amr(ref_fac=2, max_no_levs=4)
        expected = np.array([1, 2, 4, 8])
        np.testing.assert_array_equal(amr.n_ref, expected)

    def test_dx_values(self):
        amr = make_amr(n_coarse=5, ref_fac=2, max_no_levs=3, x_left=0.0, x_right=1.0)
        coarse_dx = 1.0 / (5 - 1)
        np.testing.assert_allclose(amr.dx[0], coarse_dx)
        np.testing.assert_allclose(amr.dx[1], coarse_dx / 2)
        np.testing.assert_allclose(amr.dx[2], coarse_dx / 4)

    def test_is_ref_all_false(self):
        amr = make_amr()
        assert not amr.is_ref.any()

    def test_tot_ref_lev_level0(self):
        amr = make_amr(n_coarse=5)
        assert amr.tot_ref_lev[0] == 5

    def test_x_coarse_endpoints(self):
        amr = make_amr(n_coarse=5, x_left=0.0, x_right=2.0)
        assert amr.x_coarse[0]  == pytest.approx(0.0)
        assert amr.x_coarse[-1] == pytest.approx(2.0)

    def test_f_coarse_zeros(self):
        amr = make_amr(n_coarse=5, n_comp=2)
        assert amr.f_coarse.shape == (5, 2)
        np.testing.assert_array_equal(amr.f_coarse, 0.0)

    def test_x_coord_level0_populated(self):
        amr = make_amr(n_coarse=5)
        for i in range(4):   # 4 intervals for 5 coarse points
            assert (i, 0) in amr.x_coord
            xc = amr.x_coord[(i, 0)]
            assert xc[0] == pytest.approx(amr.x_coarse[i])
            assert xc[1] == pytest.approx(amr.x_coarse[i + 1])

    def test_non_unit_domain(self):
        amr = make_amr(n_coarse=3, x_left=-1.0, x_right=1.0)
        np.testing.assert_allclose(amr.x_coarse, [-1.0, 0.0, 1.0])


# ===========================================================================
# 2. set_coarse_array / get_coarse_array
# ===========================================================================

class TestCoarseArray:
    def test_round_trip(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        vals = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        amr.set_coarse_array(vals, comp=0)
        np.testing.assert_allclose(amr.get_coarse_array(0), vals)

    def test_set_updates_f_coarse(self):
        amr = make_amr(n_coarse=4, n_comp=1)
        vals = np.ones(4) * 3.14
        amr.set_coarse_array(vals, comp=0)
        np.testing.assert_allclose(amr.f_coarse[:, 0], vals)

    def test_set_populates_f_arr_level0(self):
        amr = make_amr(n_coarse=4, n_comp=1)
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        amr.set_coarse_array(vals, comp=0)
        # Each level-0 segment should have its left-endpoint stored
        for i in range(3):
            seg = amr.f_arr.get((i, 0))
            assert seg is not None
            assert seg[0, 0] == pytest.approx(vals[i])

    def test_get_returns_copy(self):
        amr = make_amr(n_coarse=4, n_comp=1)
        vals = np.ones(4)
        amr.set_coarse_array(vals, comp=0)
        retrieved = amr.get_coarse_array(0)
        retrieved[:] = 99
        np.testing.assert_allclose(amr.f_coarse[:, 0], vals)

    def test_multicomp(self):
        amr = make_amr(n_coarse=4, n_comp=2)
        v0 = np.array([1.0, 2.0, 3.0, 4.0])
        v1 = np.array([10.0, 20.0, 30.0, 40.0])
        amr.set_coarse_array(v0, comp=0)
        amr.set_coarse_array(v1, comp=1)
        np.testing.assert_allclose(amr.get_coarse_array(0), v0)
        np.testing.assert_allclose(amr.get_coarse_array(1), v1)


# ===========================================================================
# 3. set_refinement_at / get_refinement_at
# ===========================================================================

class TestRefinementAt:
    def test_set_and_get(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        vals = _add_level1_refinement(amr, index=1)
        stored = amr.get_refinement_at(1, 1)
        assert stored is not None
        np.testing.assert_allclose(stored, vals[:amr.n_ref[1] + 1])

    def test_is_ref_flag_set(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=1)
        assert amr.is_ref[1, 0]   # level-1 refinement → level-0 flag

    def test_ref_levs_so_far_updated(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        assert amr.ref_levs_so_far == 0
        _add_level1_refinement(amr, index=1)
        assert amr.ref_levs_so_far == 1

    def test_coarse_lev_depth_updated(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=2)
        assert amr.coarse_lev_depth[2] == 1

    def test_tot_ref_lev_incremented(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        before = amr.tot_ref_lev[1]
        _add_level1_refinement(amr, index=1)
        assert amr.tot_ref_lev[1] == before + 1

    def test_get_missing_returns_none(self):
        amr = make_amr()
        assert amr.get_refinement_at(0, 1) is None

    def test_get_returns_copy(self):
        """get_refinement_at returns a copy; mutating it does not affect stored data."""
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=1)
        stored = amr.get_refinement_at(1, 1)
        assert stored is not None
        stored[0, 0] = -999
        assert amr.f_arr[(1, 1)][0, 0] != -999

    def test_level0_refinement_no_is_ref_flag(self):
        """Level-0 set_refinement_at should not set any is_ref flag."""
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3)
        vals = np.zeros((2, 1))
        amr.set_refinement_at(0, 0, vals)
        # level > 0 guard means no flag is set
        assert not amr.is_ref.any()


# ===========================================================================
# 4. Contiguous array helpers
# ===========================================================================

class TestContiguousHelpers:
    def _setup_level0(self, n_coarse=5):
        amr = make_amr(n_coarse=n_coarse, n_comp=1)
        vals = np.linspace(0, 1, n_coarse)
        amr.set_coarse_array(vals, comp=0)
        return amr, vals

    def test_get_contiguous_single_interval(self):
        amr, vals = self._setup_level0()
        result = amr.get_contiguous_refinement_array(
            level=0, comp=0, index_start=1, index_end=1
        )
        # n_ref[0]=1, so result has 2 values: vals[1] and vals[2]
        np.testing.assert_allclose(result, vals[1:3])

    def test_get_contiguous_full_span(self):
        amr, vals = self._setup_level0(n_coarse=5)
        result = amr.get_contiguous_refinement_array(
            level=0, comp=0, index_start=0, index_end=3
        )
        np.testing.assert_allclose(result, vals)

    def test_set_then_get_roundtrip(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        # Initialise level-0 f_arr slots
        amr.set_coarse_array(np.zeros(5), comp=0)
        new_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        amr.set_contiguous_refinement_array(
            level=0, comp=0, index_start=0, index_end=3, values=new_vals
        )
        result = amr.get_contiguous_refinement_array(
            level=0, comp=0, index_start=0, index_end=3
        )
        np.testing.assert_allclose(result, new_vals)

    def test_set_preserves_other_comp(self):
        amr = make_amr(n_coarse=5, n_comp=2)
        amr.set_coarse_array(np.ones(5) * 7.0, comp=0)
        amr.set_coarse_array(np.ones(5) * 3.0, comp=1)
        # overwrite comp 0 only
        amr.set_contiguous_refinement_array(
            level=0, comp=0, index_start=0, index_end=3,
            values=np.zeros(5)
        )
        result_comp1 = amr.get_contiguous_refinement_array(
            level=0, comp=1, index_start=0, index_end=3
        )
        np.testing.assert_allclose(result_comp1, np.ones(5) * 3.0)

    def test_get_x_coord_contiguous(self):
        amr = make_amr(n_coarse=5, x_left=0.0, x_right=1.0)
        result = amr.get_contiguous_refinement_x_coord(
            level=0, index_start=0, index_end=3
        )
        np.testing.assert_allclose(result, amr.x_coarse)

    def test_x_coord_length(self):
        amr = make_amr(n_coarse=5)
        result = amr.get_contiguous_refinement_x_coord(
            level=0, index_start=1, index_end=2
        )
        # 2 intervals at level-0 (n_ref=1) → 3 points
        assert len(result) == 3


# ===========================================================================
# 5. determine_refinement_segments
# ===========================================================================

class TestRefinementSegments:
    def test_single_refined_point(self):
        amr = make_amr(n_coarse=7, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=2)
        amr.determine_refinement_segments()
        assert amr.n_ref_seg[0] == 1
        assert amr.beg_ref_seg[0, 0] == 2
        assert amr.end_ref_seg[0, 0] == 2

    def test_no_refinements(self):
        amr = make_amr(n_coarse=5)
        amr.determine_refinement_segments()
        assert amr.n_ref_seg[0] == 0

    def test_two_separate_segments(self):
        amr = make_amr(n_coarse=9, n_comp=1, max_no_levs=3, ref_fac=2)
        for idx in [1, 5]:
            _add_level1_refinement(amr, index=idx)
        amr.determine_refinement_segments()
        assert amr.n_ref_seg[0] == 2


# ===========================================================================
# 6. delete_refinement_array / delete_bottom_level
# ===========================================================================

class TestDeletion:
    def test_delete_removes_entry(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=1)
        amr.delete_refinement_array(level=1, array_index=1)
        assert amr.get_refinement_at(1, 1) is None

    def test_delete_clears_is_ref(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=1)
        assert amr.is_ref[1, 0]
        amr.delete_refinement_array(level=1, array_index=1)
        assert not amr.is_ref[1, 0]

    def test_delete_decrements_tot_ref_lev(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=1)
        before = amr.tot_ref_lev[1]
        amr.delete_refinement_array(level=1, array_index=1)
        assert amr.tot_ref_lev[1] == before - 1

    def test_delete_decrements_coarse_lev_depth(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=1)
        amr.delete_refinement_array(level=1, array_index=1)
        assert amr.coarse_lev_depth[1] == 0

    def test_delete_out_of_bounds_raises(self):
        amr = make_amr(n_coarse=5)
        with pytest.raises(IndexError):
            amr.delete_refinement_array(level=0, array_index=99)

    def test_delete_empty_raises(self):
        amr = make_amr(n_coarse=5)
        with pytest.raises(RuntimeError):
            amr.delete_refinement_array(level=1, array_index=1)

    def test_delete_bottom_level_decrements_counter(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=1)
        _add_level1_refinement(amr, index=2)
        amr.delete_bottom_level()
        assert amr.ref_levs_so_far == 0

    def test_delete_bottom_level_removes_f_arr_entries(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=1)
        _add_level1_refinement(amr, index=2)
        amr.delete_bottom_level()
        assert amr.get_refinement_at(1, 1) is None
        assert amr.get_refinement_at(2, 1) is None

    def test_delete_bottom_level_clears_is_ref(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        _add_level1_refinement(amr, index=1)
        amr.delete_bottom_level()
        assert not amr.is_ref[1, 0]


# ===========================================================================
# 7. collapse_array
# ===========================================================================

class TestCollapseArray:
    def test_coarse_only_collapse(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        amr.set_coarse_array(vals, comp=0)
        x_arr, col_arr = amr.collapse_array(comp=0, start=0, finish=3)
        # Should have 4 values (left endpoints of each of 4 intervals)
        assert len(x_arr) == 4
        np.testing.assert_allclose(col_arr, vals[:4])

    def test_collapse_empty_returns_empty(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        x_arr, col_arr = amr.collapse_array(comp=0, start=0, finish=3)
        assert len(x_arr) == 0
        assert len(col_arr) == 0

    def test_refined_point_uses_fine_level(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        amr.set_coarse_array(np.zeros(5), comp=0)
        fine_vals = np.array([[10.0], [20.0], [30.0]])  # n_ref[1]+1 = 3 rows
        amr.x_coord[(1, 1)] = np.linspace(
            amr.x_coarse[1], amr.x_coarse[2], 3
        )
        amr.set_refinement_at(1, 1, fine_vals)
        _, col = amr.collapse_array(comp=0, start=1, finish=1)
        # Interval 1 is refined; first n_ref[1]=2 values should come from fine_vals
        np.testing.assert_allclose(col, fine_vals[:2, 0])


# ===========================================================================
# 8. apply_function
# ===========================================================================

class TestApplyFunction:
    def test_applies_to_coarse(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        amr.set_coarse_array(np.zeros(5), comp=0)
        result = amr.apply_function(lambda x: x ** 2)
        np.testing.assert_allclose(result.f_coarse[:, 0], amr.x_coarse ** 2)

    def test_does_not_mutate_original(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        amr.set_coarse_array(np.ones(5), comp=0)
        _ = amr.apply_function(lambda x: x * 2)
        np.testing.assert_allclose(amr.f_coarse[:, 0], np.ones(5))

    def test_applies_to_refined_level(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        amr.set_coarse_array(np.zeros(5), comp=0)
        _add_level1_refinement(amr, index=1)
        result = amr.apply_function(lambda x: np.ones_like(x) * 7.0)
        seg = result.f_arr.get((1, 1))
        assert seg is not None
        np.testing.assert_allclose(seg[:, 0], 7.0)

    def test_zero_function(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        amr.set_coarse_array(np.ones(5), comp=0)
        result = amr.apply_function(lambda x: np.zeros_like(x))
        np.testing.assert_allclose(result.f_coarse[:, 0], 0.0)


# ===========================================================================
# 9. fine_to_coarse
# ===========================================================================

class TestFineToCoarse:
    def test_fine_propagates_to_coarse_level(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        amr.set_coarse_array(np.zeros(5), comp=0)

        fine_vals = np.array([[1.0], [2.0], [3.0]])   # 3 points at level 1
        amr.x_coord[(1, 1)] = np.linspace(amr.x_coarse[1], amr.x_coarse[2], 3)
        amr.set_refinement_at(1, 1, fine_vals)

        amr.fine_to_coarse()

        # Level-0 segment at index 1 should now reflect fine boundary values
        seg0 = amr.f_arr.get((1, 0))
        assert seg0 is not None
        # With ref_fac=2: coarse point 0 of level-0 ← fine[0], point 1 ← fine[2]
        assert seg0[0, 0] == pytest.approx(fine_vals[0, 0])
        assert seg0[1, 0] == pytest.approx(fine_vals[2, 0])

    def test_returns_self(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        amr.set_coarse_array(np.zeros(5), comp=0)
        result = amr.fine_to_coarse()
        assert result is amr


# ===========================================================================
# 10. Operator overloading
# ===========================================================================

class TestOperators:
    def _simple_amr(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        amr.set_coarse_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), comp=0)
        return amr

    def test_add_scalar(self):
        amr = self._simple_amr()
        result = amr + 10
        np.testing.assert_allclose(result.f_coarse[:, 0], amr.f_coarse[:, 0] + 10)

    def test_radd_scalar(self):
        amr = self._simple_amr()
        result = 10 + amr
        np.testing.assert_allclose(result.f_coarse[:, 0], amr.f_coarse[:, 0] + 10)

    def test_sub_scalar(self):
        amr = self._simple_amr()
        result = amr - 1
        np.testing.assert_allclose(result.f_coarse[:, 0], amr.f_coarse[:, 0] - 1)

    def test_rsub_scalar(self):
        amr = self._simple_amr()
        result = 10 - amr
        np.testing.assert_allclose(result.f_coarse[:, 0], 10 - amr.f_coarse[:, 0])

    def test_mul_scalar(self):
        amr = self._simple_amr()
        result = amr * 3
        np.testing.assert_allclose(result.f_coarse[:, 0], amr.f_coarse[:, 0] * 3)

    def test_rmul_scalar(self):
        amr = self._simple_amr()
        result = 3 * amr
        np.testing.assert_allclose(result.f_coarse[:, 0], amr.f_coarse[:, 0] * 3)

    def test_truediv_scalar(self):
        amr = self._simple_amr()
        result = amr / 2
        np.testing.assert_allclose(result.f_coarse[:, 0], amr.f_coarse[:, 0] / 2)

    def test_neg(self):
        amr = self._simple_amr()
        result = -amr
        np.testing.assert_allclose(result.f_coarse[:, 0], -amr.f_coarse[:, 0])

    def test_add_amr_amr(self):
        amr = self._simple_amr()
        result = amr + amr
        np.testing.assert_allclose(result.f_coarse[:, 0], amr.f_coarse[:, 0] * 2)

    def test_sub_amr_amr(self):
        amr = self._simple_amr()
        result = amr - amr
        np.testing.assert_allclose(result.f_coarse[:, 0], 0.0)

    def test_mul_amr_amr(self):
        amr = self._simple_amr()
        result = amr * amr
        np.testing.assert_allclose(result.f_coarse[:, 0], amr.f_coarse[:, 0] ** 2)

    def test_ops_do_not_mutate_original(self):
        amr = self._simple_amr()
        original = amr.f_coarse[:, 0].copy()
        _ = amr + 100
        np.testing.assert_allclose(amr.f_coarse[:, 0], original)

    def test_ops_on_f_arr(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        amr.set_coarse_array(np.ones(5), comp=0)
        _add_level1_refinement(amr, index=1)
        result = amr * 2
        seg = result.f_arr.get((1, 1))
        orig = amr.f_arr.get((1, 1))
        assert seg is not None and orig is not None
        np.testing.assert_allclose(seg, orig * 2)

def test_unsupported_type_raises(self):
    amr = self._simple_amr()
    with pytest.raises(TypeError):
        _ = amr + "string"  # type: ignore[operator]


# ===========================================================================
# 11. __repr__
# ===========================================================================

class TestRepr:
    def test_repr_contains_key_info(self):
        amr = make_amr(n_coarse=5, n_comp=1, max_no_levs=3, ref_fac=2)
        r = repr(amr)
        assert "AMRArray" in r
        assert "n_coarse=5" in r
        assert "ref_fac=2" in r


# ===========================================================================
# 12. Edge-case / stress
# ===========================================================================

class TestEdgeCases:
    def test_single_coarse_interval(self):
        """n_coarse=2 is the minimal valid grid (one interval)."""
        amr = make_amr(n_coarse=2, n_comp=1)
        amr.set_coarse_array(np.array([0.0, 1.0]), comp=0)
        x, f = amr.collapse_array(comp=0, start=0, finish=0)
        assert len(x) == 1

    def test_ref_fac_3(self):
        amr = make_amr(n_coarse=4, n_comp=1, max_no_levs=3, ref_fac=3)
        np.testing.assert_array_equal(amr.n_ref, [1, 3, 9])

    def test_multicomp_operators(self):
        amr = make_amr(n_coarse=4, n_comp=2)
        amr.set_coarse_array(np.ones(4), comp=0)
        amr.set_coarse_array(np.ones(4) * 2, comp=1)
        result = amr + 1
        np.testing.assert_allclose(result.f_coarse[:, 0], 2.0)
        np.testing.assert_allclose(result.f_coarse[:, 1], 3.0)

    def test_deepcopy_independence(self):
        amr = make_amr(n_coarse=5, n_comp=1)
        amr.set_coarse_array(np.ones(5), comp=0)
        amr2 = copy.deepcopy(amr)
        amr2.f_coarse[:] = 99
        np.testing.assert_allclose(amr.f_coarse[:, 0], 1.0)
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any, cast
from problem_def import ProblemDef

class Burgers1D(ProblemDef):

    def initial_condition(self, x_values: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x_values, dtype=float)
        ic = np.zeros((len(x), self.n_comp), dtype=float)
        ic[:, 0] = np.exp(-((x - self.x_shift) / self.width) ** 2)
        return ic

    def eval_explicit_term(self, i_lev: int, i_seg: int, n_comp: int,
                           t: float, x_coord: NDArray[np.float64],
                           der: object, sol: NDArray[np.float64]) -> NDArray[np.float64]:
        """Explicit (nonlinear advection) term:  rhs = u * u_x."""
        D1: Any = der.sp_mat_d1[i_lev][i_seg]  # type: ignore[attr-defined]
        u_x = cast(NDArray[np.float64], D1.dot(sol))  # type: ignore[reportUnknownMemberType]
        return sol * u_x
    
    def set_bc(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.zeros(self.n_comp), np.zeros(self.n_comp)

    def compute_wave_speeds(self, amr_arr: object) -> NDArray[np.float64]:
        return amr_arr.f_coarse[:, 0:1].copy()  # type: ignore[attr-defined]
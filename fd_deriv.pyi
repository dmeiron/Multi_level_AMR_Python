import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional
from scipy.sparse import csr_matrix

class FDDeriv:
    order: int
    DN_1: NDArray[np.float64]
    DN_2: NDArray[np.float64]
    sp_mat_d1: Optional[list[list[Optional[csr_matrix]]]]
    sp_mat_d2: Optional[list[list[Optional[csr_matrix]]]]

    def form_deriv_matrices(self, amr_obj: Any) -> FDDeriv: ...

    def __init__(self, order: int) -> None: ...
    def normalize_derivatives(self, dx: float) -> None: ...
    def compute_first_derivative(
        self, n_values: int, f_values: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
    def compute_second_derivative(
        self, n_values: int, f_values: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
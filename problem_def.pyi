class ProblemDef:
    # Attributes set by __init__ kwargs
    n_comp: int
    n_coarse: int
    ref_fac: int
    max_no_levs: int
    order: int
    error_analyzer: str
    err_thr: float
    margin: int
    nu: float
    x_left: float
    x_right: float
    time_int_type: str
    cfl: float
    n_refine: int
    n_out: int
    t_end: float
    # Other attributes used across subclasses
    x_shift: float
    width: float

    def __init__(
        self,
        n_coarse: int = ...,
        n_comp: int = ...,
        ref_fac: int = ...,
        max_no_levs: int = ...,
        order: int = ...,
        error_analyzer: str = ...,
        err_thr: float = ...,
        margin: int = ...,
        nu: float = ...,
        x_left: float = ...,
        x_right: float = ...,
        time_int_type: str = ...,
        cfl: float = ...,
        n_refine: int = ...,
        n_out: int = ...,
        t_end: float = ...,
    ) -> None: ...
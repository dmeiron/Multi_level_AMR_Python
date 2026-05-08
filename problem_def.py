"""
problem_def.py
Base problem-definition class, converted from MATLAB.

Subclass ProbDef and add problem-specific parameters as needed.
The base class holds all parameters that are common across simulations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProbDef:
    """
    Encapsulates all parameters used in a simulation.

    Problem-specific subclasses should inherit from this class and add
    their own additional parameters.

    Parameters
    ----------
    n_coarse       : number of coarse grid points
    n_comp         : number of solution components
    ref_fac        : refinement factor between levels
    max_no_levs    : maximum number of refinement levels (including coarse)
    order          : polynomial order for interpolation
    error_analyzer : string indicating which error analysis to use
    err_thr        : error threshold for refinement
    margin         : number of coarse cells for refinement margin
    nu             : dissipation coefficient for all components
    x_left         : x-coordinate of left boundary
    x_right        : x-coordinate of right boundary
    time_int_type  : time integration scheme (e.g. 'imex_111')
    cfl            : CFL value for coarse mesh
    n_refine       : refine every n_refine time steps
    n_out          : produce output every n_out steps
    t_end          : integrate from t=0 to t=t_end
    """

    # Grid
    n_coarse:    int
    n_comp:      int
    ref_fac:     int
    max_no_levs: int

    # Numerics
    order:          int
    error_analyzer: str
    err_thr:        float
    margin:         int
    nu:             float

    # Domain
    x_left:  float
    x_right: float

    # Time integration
    time_int_type: str
    cfl:           float
    n_refine:      int
    n_out:         int
    t_end:         float

    def output_parameters(self) -> None:
        """Print all problem parameters to stdout."""
        print(f"  n_coarse:       {self.n_coarse}")
        print(f"  n_comp:         {self.n_comp}")
        print(f"  ref_fac:        {self.ref_fac}")
        print(f"  max_no_levs:    {self.max_no_levs}")
        print(f"  order:          {self.order}")
        print(f"  error_analyzer: {self.error_analyzer}")
        print(f"  err_thr:        {self.err_thr}")
        print(f"  margin:         {self.margin}")
        print(f"  nu:             {self.nu}")
        print(f"  x_left:         {self.x_left}")
        print(f"  x_right:        {self.x_right}")
        print(f"  time_int_type:  {self.time_int_type}")
        print(f"  cfl:            {self.cfl}")
        print(f"  n_refine:       {self.n_refine}")
        print(f"  n_out:          {self.n_out}")
        print(f"  t_end:          {self.t_end}")

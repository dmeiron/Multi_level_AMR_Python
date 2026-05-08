"""
plot_manager.py
Static plotting utilities for AMR solution visualisation, converted from MATLAB.

Key mapping notes:
  - MATLAB static methods  → Python @staticmethod
  - MATLAB persistent figure handles → module-level dict of matplotlib Figure objects
  - MATLAB tiledlayout/nexttile → matplotlib GridSpec subplots
  - MATLAB annotation textbox → matplotlib fig.text()
  - The commented-out plot_solution_comparison (Riemann problem version) is
    preserved as a comment at the bottom of the file.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.gridspec as gridspec
import numpy as np

# Module-level figure handle cache (replaces MATLAB persistent variables)
_fig_handles: dict[str, Figure] = {}


class PlotManager:

    # ------------------------------------------------------------------
    # Figure handle management
    # ------------------------------------------------------------------

    @staticmethod
    def get_figure_handle(handle_name: str) -> Figure:
        """
        Return a named Figure, creating it if it does not yet exist or has
        been closed.

        Parameters
        ----------
        handle_name : str
            One of: 'solution', 'solution_levels', 'solution_derivative',
                    'derivative_levels', 'error_levels', 'comparison'
        """
        titles = {
            'solution':            'Solution',
            'solution_levels':     'Solution Levels',
            'solution_derivative': 'Solution Derivative',
            'derivative_levels':   'Derivative Levels',
            'error_levels':        'Error Levels',
            'comparison':          'Comparison',
        }
        if handle_name not in titles:
            raise ValueError(f"Unknown figure handle name: '{handle_name}'")

        fig = _fig_handles.get(handle_name)
        # A figure is stale if it has been closed by the user
        if fig is None or not plt.fignum_exists(fig.number):
            fig = plt.figure(titles[handle_name], facecolor='w')
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(titles[handle_name])
            _fig_handles[handle_name] = fig

        return fig

    # ------------------------------------------------------------------
    # plot_solution
    # ------------------------------------------------------------------

    @staticmethod
    def plot_solution(u, t: float, n_left: int, n_right: int):
        """
        Plot all components of the collapsed AMR solution.

        Parameters
        ----------
        u       : AMRArray
        t       : float  – current time
        n_left  : int    – left coarse index (0-based)
        n_right : int    – right coarse index (0-based)
        """
        fig = PlotManager.get_figure_handle('solution')
        fig.clf()

        for i_plot in range(u.n_comp):
            ax = fig.add_subplot(u.n_comp, 1, i_plot + 1)
            x_plot, y_plot = u.collapse_array(i_plot, n_left, n_right)
            ax.plot(x_plot, y_plot, 'b--o', markersize=2)
            ax.set_title(f'Component {i_plot + 1} at time {t}')

        fig.tight_layout()
        plt.pause(0.01)

    # ------------------------------------------------------------------
    # plot_solution_levels
    # ------------------------------------------------------------------

    @staticmethod
    def plot_solution_levels(u, time: float):
        """
        Plot each component at each refinement level in a grid layout.

        Parameters
        ----------
        u    : AMRArray
        time : float – current time (used in titles)
        """
        fig = PlotManager.get_figure_handle('solution_levels')
        fig.clf()

        n_levels = u.ref_levs_so_far + 1
        comps    = u.n_comp

        gs = gridspec.GridSpec(n_levels, comps, figure=fig,
                               hspace=0.4, wspace=0.3)

        for i_comp in range(comps):
            for i_level in range(n_levels):
                ax = fig.add_subplot(gs[i_level, i_comp])
                _plot_level(ax, u, i_comp, i_level)

            # Column title above top tile
            ax_top = fig.axes[i_comp]   # first axes in this column
            ax_top.set_title(
                f'Component {i_comp + 1} levels at time {time}',
                fontsize=10
            )

        plt.pause(0.01)

    # ------------------------------------------------------------------
    # plot_solution_derivative
    # ------------------------------------------------------------------

    @staticmethod
    def plot_solution_derivative(u_x, t: float, n_left: int, n_right: int):
        """Plot all components of the derivative array."""
        fig = PlotManager.get_figure_handle('solution_derivative')
        fig.clf()

        for i_plot in range(u_x.n_comp):
            ax = fig.add_subplot(u_x.n_comp, 1, i_plot + 1)
            x_plot, y_plot = u_x.collapse_array(i_plot, n_left, n_right)
            ax.plot(x_plot, y_plot, 'b--o', markersize=2)
            ax.set_title(f'Component {i_plot + 1} derivative at time {t}')

        fig.tight_layout()
        plt.pause(0.01)

    # ------------------------------------------------------------------
    # plot_derivative_levels
    # ------------------------------------------------------------------

    @staticmethod
    def plot_derivative_levels(u_x, time: float):
        """Plot derivative components at each refinement level."""
        fig = PlotManager.get_figure_handle('derivative_levels')
        fig.clf()

        n_levels = u_x.ref_levs_so_far + 1
        comps    = u_x.n_comp

        gs = gridspec.GridSpec(n_levels, comps, figure=fig,
                               hspace=0.4, wspace=0.3)

        for i_comp in range(comps):
            for i_level in range(n_levels):
                ax = fig.add_subplot(gs[i_level, i_comp])
                _plot_level(ax, u_x, i_comp, i_level)

            ax_top = fig.axes[i_comp]
            ax_top.set_title(
                f'Component derivative {i_comp + 1} levels at time {time}',
                fontsize=10
            )

        plt.pause(0.01)

    # ------------------------------------------------------------------
    # plot_error_levels
    # ------------------------------------------------------------------

    @staticmethod
    def plot_error_levels(err, time: float):
        """Plot error indicator components at each refinement level."""
        fig = PlotManager.get_figure_handle('error_levels')
        fig.clf()

        n_levels = err.ref_levs_so_far + 1
        comps    = err.n_comp

        gs = gridspec.GridSpec(n_levels, comps, figure=fig,
                               hspace=0.4, wspace=0.3)

        for i_comp in range(comps):
            for i_level in range(n_levels):
                ax = fig.add_subplot(gs[i_level, i_comp])
                _plot_level(ax, err, i_comp, i_level)

            ax_top = fig.axes[i_comp]
            ax_top.set_title(
                f'Component error {i_comp + 1} levels at time {time}',
                fontsize=10
            )

        plt.pause(0.01)

    # ------------------------------------------------------------------
    # plot_solution_comparison  (numerical vs analytic)
    # ------------------------------------------------------------------

    @staticmethod
    def plot_solution_comparison(u, pd, t: float, n_left: int, n_right: int):
        """
        Compare numerical and analytic solutions (primitive variables).

        Requires pd to expose:
          .conserved_to_primitive(mass, momentum, energy) -> (rho, u_vel, press)
          .rho_l, .u_l, .p_l, .rho_r, .u_r, .p_r
          .x_left, .x_right, .x_diaphragm
        and a callable euler_riemann_problem() importable from the problem module.
        """
        from euler_riemann_problem import euler_riemann_problem  # problem-specific import

        fig = PlotManager.get_figure_handle('comparison')
        fig.clf()

        x_plot,  mass     = u.collapse_array(0, n_left, n_right)
        _,       momentum = u.collapse_array(1, n_left, n_right)
        _,       energy   = u.collapse_array(2, n_left, n_right)

        rho, u_vel, press = pd.conserved_to_primitive(mass, momentum, energy)

        npts = len(x_plot)
        rho_a, u_vel_a, p_a = euler_riemann_problem(
            pd.rho_l, pd.u_l, pd.p_l,
            pd.rho_r, pd.u_r, pd.p_r,
            t, pd.x_left, pd.x_right, pd.x_diaphragm,
            x_plot, npts
        )

        pairs = [
            (rho,   rho_a,   'density'),
            (u_vel, u_vel_a, 'velocity'),
            (press, p_a,     'pressure'),
        ]

        gs = gridspec.GridSpec(u.n_comp, 2, figure=fig, hspace=0.4, wspace=0.3)

        for k, (num, ana, label) in enumerate(pairs):
            ax_sol = fig.add_subplot(gs[k, 0])
            ax_sol.plot(x_plot, num, 'b--o', markersize=2, label='numerical')
            ax_sol.plot(x_plot, ana, label='analytic')
            ax_sol.set_title(f'Component {k+1} ({label}) at time {t}')
            ax_sol.legend(fontsize=7)

            ax_err = fig.add_subplot(gs[k, 1])
            ax_err.plot(x_plot, num - ana, 'b--o', markersize=2)
            ax_err.set_title(f'Component {k+1} error at time {t}')

        plt.pause(0.01)


# ------------------------------------------------------------------
# Module-level helper (not a class method — mirrors MATLAB's nested
# plot_level function used inside plot_*_levels methods)
# ------------------------------------------------------------------

def _plot_level(ax: Axes, u, i_comp: int, i_level: int):
    """
    Plot the data for component i_comp at refinement level i_level on ax.
    Data is gathered by scanning the f_arr dict for entries at this level.
    Both i_comp and i_level are 0-indexed.
    """
    x_vals = []
    f_vals = []

    for i_coarse in range(u.n_coarse - 1):
        if u.coarse_lev_depth[i_coarse] >= i_level:
            xc  = u.x_coord.get((i_coarse, i_level))
            arr = u.f_arr.get((i_coarse, i_level))
            if xc is not None and arr is not None:
                n_r = u.n_ref[i_level]
                x_vals.append(xc[:n_r])
                f_vals.append(arr[:n_r, i_comp])

    if x_vals:
        x_all = np.concatenate(x_vals)
        f_all = np.concatenate(f_vals)
        ax.plot(x_all, f_all, 'b-o', markersize=2)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    ax.set_xlabel('x', fontsize=7)
    ax.tick_params(labelsize=7)

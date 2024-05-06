import numpy as np

from plot import init_plot, update_plot, save_plot


class Model:

    def __init__(self, params, *args, **kwargs):
        [setattr(self, k, v) for k, v in params.items()]

        self.x, dx = np.linspace(*self.x_range, self.N_x + 1, retstep=True)
        self.y, dy = np.linspace(*self.y_range, self.N_y + 1, retstep=True)
        self.dX = np.array([dx, dy])

        self.t, self.dt = np.linspace(*self.t_range, self.N_t + 1,
                                      retstep=True)

        self.x_grid, self.y_grid = np.meshgrid(self.x, self.y)

        self.current_time_step = 0

        # Plotting
        if not self.plot or self.plot is None:
            return

        self._fig = None
        self._ax = None
        self._ax_plot = None
        self._writer = None

    def initialize(self, func=None, *args, **kwargs):
        if func is not None:
            func(self, *args, **kwargs)

        # Plotting
        if self.plot:
            init_plot(self)

    def full_run(self, *args, **kwargs):
        for _ in range(self.N_t):
            # break
            self.update(*args, **kwargs)
            self.current_time_step += 1

            if self.plot and self.animate:
                update_plot(self)

        # Plotting
        if self.plot:
            save_plot(self)

    def __str__(self):
        text = (
            f'x bounds    : {self.x_range}\n'
            f'y bounds    : {self.y_range}\n'
            f'cell width  : {self.dX}\n'
            f'time domain : {self.t_range}\n'
            f'time step   : {self.dt}\n'
        )
        return text

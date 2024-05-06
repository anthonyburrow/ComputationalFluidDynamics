import numpy as np

from Model import Model
from InitialCondition import square_func, constant_bound
from plot import _plot_func_params


class Diffusion(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.u = np.zeros((self.N_y + 1, self.N_x + 1))

        self.r = self.nu * self.dt / self.dX**2

    def update(self, method=None, periodic=True):
        if method is None:
            method == 'ftcs'

        if method == 'ftcs':
            self._ftcs_step()

    def _ftcs_step(self):
        self.u[1:-1, 1:-1] += \
            self.r[0] * (self.u[1:-1, 2:] - 2. * self.u[1:-1, 1:-1] + self.u[1:-1, :-2]) + \
            self.r[1] * (self.u[2:, 1:-1] - 2. * self.u[1:-1, 1:-1] + self.u[:-2, 1:-1])

        constant_bound(self.u, 1.)

    def _plot_func(self):
        sc = self._ax.contourf(self.x_grid, self.y_grid, self.u,
                               **_plot_func_params)
        return sc


params = {
    'N_x': 30,                                  # x resolution
    'x_range': (0., 2.),                        # x range
    'N_y': 30,                                  # y resolution
    'y_range': (0., 2.),                        # y range
    'N_t': 20,                                  # Time steps
    't_range': (0., 0.4),                       # Time range
    'nu': 0.05,                                 #
    # Plot properties
    'plot': True,                               # Control all plotting
    'plot_fn': './Diffusion.pdf',               # Output plot filename
    'animate': True,                            # Save frames as animation
    'animation_frames_to_skip': 5,              # Save every nth frame
    'gif_fn': './Diffusion.gif',                # Output gif filename
}


if __name__ == '__main__':
    model = Diffusion(params)
    model.initialize(square_func)

    model.full_run(method='ftcs')

    print(model)

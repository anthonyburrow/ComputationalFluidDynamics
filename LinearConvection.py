import numpy as np

from Model import Model
from InitialCondition import square_func, constant_bound
from plot import _plot_func_params


class LinearConvection(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.u = np.zeros((self.N_y + 1, self.N_x + 1))

        self.C = self.c * self.dt / self.dX

    def update(self, method=None, periodic=True):
        if method is None:
            method == 'upstream'

        if method == 'upstream':
            self._upstream_step()

    def _upstream_step(self):
        self.u[1:, 1:] -= \
            self.C[0] * (self.u[1:, 1:] - self.u[1:, :-1]) + \
            self.C[1] * (self.u[1:, 1:] - self.u[:-1, 1:])

        constant_bound(self.u, 1.)

    def _plot_func(self):
        sc = self._ax.contourf(self.x_grid, self.y_grid, self.u,
                               **_plot_func_params)
        return sc


params = {
    'N_x': 80,                                  # x resolution
    'x_range': (0., 2.),                        # x range
    'N_y': 80,                                  # y resolution
    'y_range': (0., 2.),                        # y range
    'N_t': 100,                                 # Time steps
    't_range': (0., 0.5),                       # Time range
    'c': 1.,                                    # Constant phase speed
    # Plot properties
    'plot': True,                               # Control all plotting
    'plot_fn': './LinearConvection.pdf',        # Output plot filename
    'animate': True,                            # Save frames as animation
    'animation_frames_to_skip': 5,              # Save every nth frame
    'gif_fn': './LinearConvection.gif',         # Output gif filename
}


if __name__ == '__main__':
    model = LinearConvection(params)
    model.initialize(square_func)

    model.full_run(method='upstream')

    print(model)

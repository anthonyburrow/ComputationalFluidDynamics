import numpy as np

from Model import Model
from InitialCondition import constant_bound
from plot import _plot_func_params


class Cavity(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.u = np.zeros((self.N_y + 1, self.N_x + 1, 2))
        self.p = np.zeros((self.N_y + 1, self.N_x + 1))
        self.b = np.zeros((self.N_y + 1, self.N_x + 1))

        self.C = self.dt / self.dX
        self.r = self.nu * self.dt / self.dX**2

    def update(self):
        self._pressure_step()
        self._velocity_step()

    def _velocity_step(self):
        self.u[1:-1, 1:-1, 0] += \
            -self.C[0] * self.u[1:-1, 1:-1, 0] * (self.u[1:-1, 1:-1, 0] - self.u[1:-1, :-2, 0]) + \
            -self.C[1] * self.u[1:-1, 1:-1, 1] * (self.u[1:-1, 1:-1, 0] - self.u[:-2, 1:-1, 0]) + \
            -(0.5 * self.C[0] / self.rho) * (self.p[1:-1, 2:] - self.p[1:-1, :-2]) + \
            self.r[0] * (self.u[1:-1, 2:, 0] - 2. * self.u[1:-1, 1:-1, 0] + self.u[1:-1, :-2, 0]) + \
            self.r[1] * (self.u[2:, 1:-1, 0] - 2. * self.u[1:-1, 1:-1, 0] + self.u[:-2, 1:-1, 0])

        self.u[1:-1, 1:-1, 1] += \
            -self.C[0] * self.u[1:-1, 1:-1, 0] * (self.u[1:-1, 1:-1, 1] - self.u[1:-1, :-2, 1]) + \
            -self.C[1] * self.u[1:-1, 1:-1, 1] * (self.u[1:-1, 1:-1, 1] - self.u[:-2, 1:-1, 1]) + \
            -(0.5 * self.C[1] / self.rho) * (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) + \
            self.r[0] * (self.u[1:-1, 2:, 1] - 2. * self.u[1:-1, 1:-1, 1] + self.u[1:-1, :-2, 1]) + \
            self.r[1] * (self.u[2:, 1:-1, 1] - 2. * self.u[1:-1, 1:-1, 1] + self.u[:-2, 1:-1, 1])

        self.u[0, :, 0] = 0.
        self.u[:, 0, 0] = 0.
        self.u[:, -1, 0] = 0.
        self.u[-1, :, 0] = 1.
        constant_bound(self.u[:, :, 1], 0.)

    def _calc_b(self):
        self.b[1:-1, 1:-1] = \
            (1. / self.dt) * (
                (self.u[1:-1, 2:, 0] - self.u[1:-1, :-2, 0]) / self.dX[0] + \
                (self.u[2:, 1:-1, 1] - self.u[:-2, 1:-1, 1]) / self.dX[1]
            ) - \
            0.5 * ((self.u[1:-1, 2:, 0] - self.u[1:-1, :-2, 0]) / self.dX[0])**2 - \
            (1. / (self.dX[0] * self.dX[1])) * (
                (self.u[2:, 1:-1, 0] - self.u[:-2, 1:-1, 0]) * \
                (self.u[1:-1, 2:, 1] - self.u[1:-1, :-2, 1])
            ) - \
            0.5 * ((self.u[2:, 1:-1, 1] - self.u[:-2, 1:-1, 1]) / self.dX[1])**2

        self.b[1:-1, 1:-1] *= 0.5 * self.rho

    def _pressure_step(self):
        self._calc_b()

        self.p[1:-1, 1:-1] = \
            (self.p[1:-1, 2:] + self.p[1:-1, :-2]) * self.dX[1]**2 + \
            (self.p[2:, 1:-1] + self.p[:-2, 1:-1]) * self.dX[0]**2 - \
            self.dX[0]**2 * self.dX[1]**2 * self.b[1:-1, 1:-1]

        self.p[1:-1, 1:-1] *= 0.5 / (self.dX @ self.dX)

        self.p[:, -1] = self.p[:, -2]
        self.p[0, :] = self.p[1, :]
        self.p[:, 0] = self.p[:, 1]
        self.p[-1, :] = 0.

    def _plot_func(self):
        self._ax.quiver(self.x_grid[::2, ::2], self.y_grid[::2, ::2],
                        self.u[::2, ::2, 0], self.u[::2, ::2, 1], zorder=1)

        sc = self._ax.contourf(self.x_grid, self.y_grid, self.p,
                               **_plot_func_params)
        return sc


params = {
    'N_x': 40,                                  # x resolution
    'x_range': (0., 2.),                        # x range
    'N_y': 40,                                  # y resolution
    'y_range': (0., 2.),                        # y range
    'N_t': 300,                                 # Time steps
    't_range': (0., 0.3),                       # Time range
    'nu': 0.1,                                  #
    'rho': 1.,                                  # Density
    # Plot properties
    'plot': True,                               # Control all plotting
    'plot_fn': './NavierStokes_Cavity.pdf',     # Output plot filename
    'animate': True,                            # Save frames as animation
    'animation_frames_to_skip': 5,              # Save every nth frame
    'gif_fn': './NavierStokes_Cavity.gif',      # Output gif filename
}


if __name__ == '__main__':
    model = Cavity(params)
    model.initialize()

    model.full_run()

    print(model)

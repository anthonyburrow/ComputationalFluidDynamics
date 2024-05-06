import numpy as np


def square_func(model):
    '''Makes a square'''
    if hasattr(model, 'u') and model.u is not None:
        model.u[:, :] = 1.
        model.u[int(0.25 * model.N_y):int(0.5 * model.N_y + 1),
                int(0.25 * model.N_x):int(0.5 * model.N_x + 1)] = 2.
    if hasattr(model, 'v') and model.v is not None:
        model.v[:, :] = 1.
        model.v[int(0.25 * model.N_y):int(0.5 * model.N_y + 1),
                int(0.25 * model.N_x):int(0.5 * model.N_x + 1)] = 2.


def constant_bound(arr, const):
    arr[0, :] = const
    arr[-1, :] = const
    arr[:, 0] = const
    arr[:, -1] = const


def step_func(model):
    '''Makes a step function'''
    x = model.x
    t = model.t_range[0]
    u = model.u

    u[:] = 0.

    x0 = 0.5 * (1. + t)

    mask = x < x0
    u[mask] = 1.

    u[x == x0] = 0.5


def gaussian_func(model, mu=1., sig=1. / 8.):
    '''Makes a gaussian function'''
    x = model.x
    t = model.t_range[0]
    u = model.u
    nu = model.nu

    var = 2. * nu * t + sig**2
    A = 1. / np.sqrt(2. * np.pi * var)

    u[:] = A * np.exp(-0.5 * (x - mu)**2 / var)

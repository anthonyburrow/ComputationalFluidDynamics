import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib import cm, colors


_gif_fps = 15
_gif_metadata = {
    'title': 'Hydro',
}

_default_plot_fn = './hydro.png'
_default_gif_fn = './hydro.gif'

_norm_range = (-3., 3.)

_plot_func_params = {
    'levels': np.linspace(*_norm_range, 11),
    'norm': colors.Normalize(*_norm_range),
    'cmap': cm.viridis,
    'antialiased': True,
    'zorder': 0,
}


def update_plot(model):
    for artist in plt.gca().lines + plt.gca().collections:
        artist.remove()

    t_ind = model.current_time_step

    if model.animate:
        is_skipped_frame = t_ind % model.animation_frames_to_skip
        if is_skipped_frame:
            return

    sc = model._plot_func()

    t = model.t[t_ind]
    model._ax.set_title(f'time = {t:.3f}')

    if model.animate:
        model._writer.grab_frame()

    return sc


def init_anim(model):
    # Setup animation
    fn = model.gif_fn if model.gif_fn is not None else _default_gif_fn

    model._writer = PillowWriter(fps=_gif_fps, metadata=_gif_metadata)
    model._writer.setup(model._fig, fn)

    # Plot initial condition
    sc = update_plot(model)
    model._fig.colorbar(sc, ax=model._ax)


def init_plot(model):
    model._fig, model._ax = plt.subplots()

    # Plot properties
    model._ax.set_xlabel('x')
    model._ax.set_ylabel('y')

    model._ax.set_xlim(*model.x_range)
    model._ax.set_xlim(*model.y_range)

    if model.animate:
        init_anim(model)


def save_plot(model):
    # plt.tight_layout()

    update_plot(model)

    fn = model.plot_fn if model.plot_fn is not None else _default_plot_fn
    model._fig.savefig(fn)

    if model.animate:
        model._writer.finish()

    plt.close('all')

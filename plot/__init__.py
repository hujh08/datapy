# module to plot data

from .layout import RectManager
from .figax import *
from .params import *

from .colors import *

from .plot import *
from .markline import *
from .curvefit import *
from .patch import *

from .tools import *

# starting point for layout
def init(*args, **kwargs):
    '''
        start for layout

        return RectManager object
    '''
    return RectManager(*args, **kwargs)

def subplots(*args, origin_upper=True, **kwargs):
    '''
        similar as `plt.subplots` with origin rect in upper
        
        different is with additional args for distances' ratio
            see `get_figaxes_grid` for detail
    '''
    return get_figaxes_grid(*args, origin_upper=origin_upper, **kwargs)

def split_axes(axes, *args, loc=[0, 1], replace=True,
                            origin_upper=True, **kwargs):
    '''
        split axes into grid of axes
            wrap of `get_figaxes_in_axes`

        change default kws:
            loc: default [0, 1]
            replace: default True
            origin_upper: default True
    '''
    kwargs.update(loc=loc, replace=replace, origin_upper=origin_upper)
    return get_figaxes_in_axes(axes, *args, **kwargs)

def join_axes(axes, *axs, nrows=1, ncols=1, loc=[0, 1], replace=True,
                        origin_upper=True, **kwargs):
    '''
        join multiply axes to one axes or a grid

        new axes/grid would cover extent of all given axes
    '''
    kwargs.update(loc=loc, origin_upper=origin_upper, nrows=nrows, ncols=ncols)

    axs=(axes,)+axs
    axes_res=get_figaxes_grid(at=axs, **kwargs)

    if replace:
        for ax in axs:
            ax.remove()

    return axes_res

def show():
    '''
        wrapper of plt.show
    '''
    import matplotlib.pyplot as plt
    plt.show()
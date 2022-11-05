# module to plot data

from .layout import RectManager
from .figax import *
from .params import *

from .plot import *
from .markline import *
from .curvefit import *

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

def show():
    '''
        wrapper of plt.show
    '''
    import matplotlib.pyplot as plt
    plt.show()
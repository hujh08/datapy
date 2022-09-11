# module to plot data

from .layout import RectManager
from .plot import *
from .figax import *

# starting point for layout
def init(*args, **kwargs):
    '''
        start for layout

        return RectManager object
    '''
    return RectManager(*args, **kwargs)

def show():
    '''
        wrapper of plt.show
    '''
    import matplotlib.pyplot as plt
    plt.show()
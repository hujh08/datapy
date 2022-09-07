#!/usr/bin/env python3

'''
    simple and useful functions to plot
'''

import numpy as np
import matplotlib.lines as mlines

from .transforms import yFuncTransFormFromAxes

# decorate

## mark line
def add_fcurve(ax, func, xscope=[0, 1], npoints=100, **kwargs):
    '''
        add function curve
    '''
    # transformation
    trans = yFuncTransFormFromAxes(ax, func)

    # points
    x0, x1=xscope
    xs=np.linspace(x0, x1, npoints)

    # construct line
    l=mlines.Line2D(xs, xs, transform=trans, **kwargs)
    ax.add_line(l)

    return l

def add_line(ax, k, b, **kwargs):
    '''
        add line: k*x + b
    '''
    func=lambda x: k*x+b
    return add_fcurve(ax, func, **kwargs)

def add_dline(ax, offset=0, **kwargs):
    '''
        add diagnoal line with optional offset
    '''
    return add_line(ax, 1, offset, **kwargs)

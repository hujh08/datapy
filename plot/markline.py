#!/usr/bin/env python3

'''
    add line to mark tendence in plot
'''

import numpy as np
from .transforms import yFuncTransFormFromAxes

__all__=['add_fcurve', 'add_line', 'add_dline']

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
    l,=ax.plot(xs, xs, transform=trans, **kwargs)

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

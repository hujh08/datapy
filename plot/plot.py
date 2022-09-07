#!/usr/bin/env python3

'''
    simple and useful functions to plot
'''

import numpy as np
import matplotlib.lines as mlines

from .transforms import yFuncTransFormFromAxes

# distribution plot

def plot_hist(ax, xs, bins=50, xlim=None, density=True, histtype='step', **kwargs):
    '''
        histogram plot

        hujh-friendly kwargs
    '''
    return ax.hist(xs, bins=bins, density=density, histtype=histtype,
                range=xlim, **kwargs)

def plot_cumul(ax, xs, xlim=None, **kwargs):
    '''
        cumulative plot
    '''
    xs=np.asanyarray(xs)

    # limit cut
    if xlim is not None:
        x0, x1=xlim

        if x0 is not None:
            xs=xs[xs>=x0]

        if x1 is not None:
            xs=xs[xs<=x1]

    # sort
    n=len(xs)
    fracs=np.arange(n+1)/n
    
    xsorts=np.sort(xs)
    xsorts=[xsorts[0], *xsorts]  # append zero point

    # plot
    return ax.step(xsorts, fracs, where='post', **kwargs)

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

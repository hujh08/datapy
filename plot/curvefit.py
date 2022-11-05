#!/usr/bin/env python3

'''
    curve fit to histogram
'''

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.patches as mpatch
import matplotlib.container as mcontainer
from matplotlib.lines import Line2D as mline

from .markline import add_fcurve

__all__=['add_gauss_fit']

# gaussian fit
def gauss_1d(x, x0, sigma, I):
    return I*np.exp(-(x-x0)**2/(2*sigma**2))

def cents_to_edges(cents):
    '''
        bin center to bin edges
    '''
    semiws=np.diff(cents)/2
    edges=cents[1:]-semiws
    edges=np.asarray([cents[0]-semiws[0], *edges, cents[-1]+semiws[-1]])

    return edges

def fit_gauss1d_to_data(cnts, xs):
    if len(cnts)+1==len(xs):
        edges=xs
        cents=(edges[:-1]+edges[1:])/2
    elif len(cnts)==len(xs):
        cents=xs
        edges=cents_to_edges(cents)
    else:
        raise ValueError('mismatch between len of `cnts` and `xs`')

    # init guess
    ws=cnts/np.sum(cnts)
    x0=np.sum(ws*cents)
    std=np.sqrt(np.sum(ws*(cents-x0)**2))
    I=np.sum(cnts*np.diff(edges))/(np.sqrt(2*np.pi)*std)
    p0=(x0, std, I)

    popt, _=curve_fit(gauss_1d, cents, cnts, p0=p0)

    func=lambda x: gauss_1d(x, *popt)

    return func, popt

def get_data_from_polygon(p):
    '''
        get cnts, edges from object returned from `hist` plot
    '''
    path=p.get_path()
    verts=path.vertices

    xs, ys=verts.T

    # stepfilled
    backs,=np.nonzero(np.diff(xs)<0)
    if len(backs)>0:
        n=backs[0]+1
        xs=xs[:n]
        ys=ys[:n]

    cnts=ys[1:-1:2]
    edges=xs[::2]

    return cnts, edges

def get_data_from_line(l):
    '''
        get ys, xs from Line2D
    '''
    assert isinstance(l, mline)
    xs, ys=l.get_data()

    return ys, xs

def get_data_from_bars(p):
    '''
        cnts, edges from BarContainer
    '''
    cnts=[]
    cents=[]
    # edges=[]
    for b in p:
        x0, y0=b.get_xy()
        w=b.get_width()
        h=b.get_height()

        cnts.append(y0+h)
        cents.append(x0+w/2)

    cnts=np.asarray(cnts)
    cents=np.asarray(cents)

    # bin centers to edges
    edges=cents_to_edges(cents)

    return cnts, edges

def get_data_from_plt(p):
    '''
        get cnts, edges from object returned from `hist` plot
    '''
    if isinstance(p, list):
        p=p[0]

    if isinstance(p, mpatch.Polygon):
        return get_data_from_polygon(p)

    assert isinstance(p, mcontainer.BarContainer), \
           'only support `mpatch.Polygon` and `mcontainer.BarContainer`'
    return get_data_from_bars(p)

def add_gauss_fit(*args, **kwargs):
    '''
        add gaussian fit for hist plot

        2 way to call
            add_gauss_fit(p, **kwargs)
            add_gauss_fit(ax, cnts, edges)
    '''
    if len(args)==1:
        p,=args
        if isinstance(p, list):
            p=p[0]
    
        if isinstance(p, mpatch.Polygon):
            ax=p.axes
        elif isinstance(p, mcontainer.BarContainer):
            ax=p[0].axes
        else:
            s='only support `mpatch.Polygon` and `mcontainer.BarContainer`'
            raise TypeError(s)
    
        cnts, edges=get_data_from_plt(p)
    else:
        ax, cnts, edges=args

    func, popt=fit_gauss1d_to_data(cnts, edges)

    add_fcurve(ax, func, **kwargs)

    return popt

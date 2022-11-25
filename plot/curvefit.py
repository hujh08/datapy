#!/usr/bin/env python3

'''
    curve fit to histogram
'''

import collections

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.axes as maxes
import matplotlib.patches as mpatches
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

    # to namedtuple
    t_gauss1d=collections.namedtuple('Gauss1d', ['x0', 'sigma', 'I'])
    popt=t_gauss1d(*popt)

    return func, popt

# data from object returned by hist plot
def get_data_from_polygon(p):
    '''
        get cnts, edges from object returned from `hist` plot
    '''
    path=p.get_path()
    verts=path.vertices

    xs, ys=verts.T

    # stepfilled
    backs,=np.nonzero(np.diff(xs)<0)  # backward path
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
    if isinstance(p, mpatches.Polygon):
        return get_data_from_polygon(p)

    # list returned from hist plot
    if len(p)==1 and isinstance(p[0], mpatches.Polygon):
        return get_data_from_polygon(p[0])

    # bar collection
    if not all([isinstance(t, mpatches.Rectangle) for t in p]):
        s='only support `mpatches.Polygon` and collection of bars'
        raise ValueError(s)

    return get_data_from_bars(p)

# get patches from ax
def split_hist_patches(patches):
    '''
        split hist patches based on
            - type: polygon (for step) and rectangle (for bars)
            - fc: facecolor for bars
    '''
    hists=[]
    prevfc=None  # fc of previous patch, None if not bar
    for p in patches:
        if isinstance(p, mpatches.Polygon):
            hists.append([p])
            prevfc=None
            continue
        elif not isinstance(p, mpatches.Rectangle):
            # only consider Polygon and Rectangle
            continue

        # first bar in new group
        if prevfc is None or p.get_fc()!=prevfc:
            hists.append([p])
            prevfc=p.get_fc()
        else:  # same group
            hists[-1].append(p)

    return hists

def get_patches_from_ax(ax, hlabel=None, hind=None):
    '''
        get patches of hist plot from given ax

        patches in ax is first splitted to groups of hist plot,
        based on
            - type: polygon (for step) and rectangle (for bars)
            - fc: facecolor for bars

        if `hlabel` is given, groups with given label is selected

        `hind` specify index of group in hists to return

        if both `hlabel` and `hind` None, use all patches
    '''
    if hlabel is None and hind is None:
        return ax.patches

    hists=split_hist_patches(ax.patches)
    if hlabel is not None:
        hists=[g for g in hists if g[0].get_label()==hlabel]

    if hind is None:
        if len(hists)>1:
            raise ValueError('too many hist groups found. use `hind` to specify one')

        return hists[0]

    return hists[hind]

def add_gauss_fit(*args, **kwargs):
    '''
        add gaussian fit for hist plot

        2 way to call
            add_gauss_fit(p, **kwargs)  # for p from hist plot
            add_gauss_fit(ax, hlabel='some hist', hind=0) # use patches with given label in ax
            add_gauss_fit(ax, cnts, edges)
    '''
    if len(args)==1:
        p,=args
        if isinstance(p, maxes.Axes):
            ax=p

            pkws={}
            for k in ['hlabel', 'hind']:
                if k in kwargs:
                    pkws[k]=kwargs.pop(k)
            p=get_patches_from_ax(ax, **pkws)

        elif isinstance(p, mpatches.Polygon):
            ax=p.axes
        else:
            ax=p[0].axes
    
        cnts, edges=get_data_from_plt(p)
    else:
        ax, cnts, edges=args

    func, popt=fit_gauss1d_to_data(cnts, edges)

    add_fcurve(ax, func, **kwargs)

    return popt

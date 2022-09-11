#!/usr/bin/env python3

'''
    simple and useful functions to plot
'''

import numpy as np
import numbers

from matplotlib.colors import LinearSegmentedColormap

from .transforms import yFuncTransFormFromAxes
from ._tools import filter_by_lim, calc_density_map_2d, quants_to_levels
from .colors import get_next_color_in_cycle
from .legend import handler_nonfill, update_handler_for_contour

__all__=['plot_hist', 'plot_cumul',
         'plot_2d_hist', 'plot_2d_contour', 'plot_2d_scatter',
         'plot_2d_joint',
         'add_fcurve', 'add_line', 'add_dline']

# distribution plot

## 1d
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
    xs=filter_by_lim(xs, xlim=xlim)

    # sort
    n=len(xs)
    fracs=np.arange(n+1)/n
    
    xsorts=np.sort(xs)
    xsorts=[xsorts[0], *xsorts]  # append zero point

    # plot
    return ax.step(xsorts, fracs, where='post', **kwargs)

## 2d

def plot_2d_hist(ax, *args, bins=None,
                    xlim=None, ylim=None, limtype='v',
                    kde=False, qthresh=None,
                    origin='lower', aspect='auto', **kwargs):
    '''
        2d hist plot

        Parameters:
            args: (xs, ys) or (dens, xcents, ycents)
                data to plot

            bins, xlim, ylim, limtype: kwargs to calculate density map

            qthresh: None or float in [0, 1]
                threshold in quantile
                cells in this quantile would be transparent

        Some parameters have default changed:
            origin: default: 'lower'
            
            aspect: default: 'auto'
                axes is kept fixed, maybe resulting non-square pixels
                BUT square pixels don't make sense for 2d distribution
    '''
    if len(args)==2:
        xs, ys=args
        # calculate histogram
        dens, (xcents, ycents)=\
            calc_density_map_2d(xs, ys, bins=bins, kde=kde,
                        xlim=xlim, ylim=ylim, limtype=limtype)
    elif len(args)==3:
        dens, xcents, ycents=args
    else:
        raise ValueError('only allow (xs, ys) or (dens, xcents, ycents) as arguments')

    # extent of hist
    dxs, dys=np.diff(xcents), np.diff(ycents)

    x0, x1=xcents[0]-0.5*dxs[0], xcents[-1]-0.5*dxs[-1]
    y0, y1=ycents[0]-0.5*dys[0], ycents[-1]-0.5*dys[-1]

    # thresh
    if qthresh is not None:
        assert 0<=qthresh<=1
        thresh=quants_to_levels(dens, qthresh)
        # dens[dens<thresh]=np.nan
        dens=np.ma.masked_less_equal(dens, thresh)

    return ax.imshow(dens, extent=[x0, x1, y0, y1],
                     origin=origin, aspect=aspect, **kwargs)

def plot_2d_contour(ax, *args, kde=False, bins=None,
                    xlim=None, ylim=None, limtype='v',
                    levels=None, qthresh=0, **kwargs):
    '''
        (non-filled) contour plot

        Paramters:
            args: (xs, ys) or (dens, xcents, ycents)
                data to plot

            kde: bool
                whether use KDE to compute density map

            bins, xlim, ylim, limtype: kwargs to calculate density map

            levels: None, int, or array of numbers
                quantiles of contours to plot

                if None: use levels=10
                if int: use `np.linspace(qthresh, 1, levels+1)[:-1]`
                    or `np.linspace(0, 1, levels+1)[1:-1]` if qthresh==0

            qthresh: float
                threshold of contour to plot

        Optional Parameters:
            color(s), linewidth(s) (or lw), linestyle(s) (or ls):
                setup for line of contours
                
                if singular keyword given, use same setup for all contours

                if not `color`(s) given, use color cycle

            label: str
                label for contour
    '''
    if len(args)==2:
        xs, ys=args
        # calculate histogram
        dens, (xcents, ycents)=\
            calc_density_map_2d(xs, ys, bins=bins, kde=kde,
                        xlim=xlim, ylim=ylim, limtype=limtype)
    elif len(args)==3:
        dens, xcents, ycents=args
    else:
        raise ValueError('only allow (xs, ys) or (dens, xcents, ycents) as arguments')

    # levels
    if levels is None:
        levels=10
    if isinstance(levels, numbers.Number):
        assert 0<=qthresh<=1
        levels=np.linspace(qthresh, 1, levels+1)[:-1]
        if qthresh==0:
            levels=levels[1:]
    else:
        assert all([0<=l<=1 for l in levels])
        levels=np.sort(levels)

    ## to levels in density map
    vlevels=quants_to_levels(dens, levels)

    # setup of contour's lines
    ## color
    if all([k not in kwargs for k in ['color', 'colors']]):
        kwargs['color']=None  # default color

    if 'color' in kwargs:
        assert 'colors' not in kwargs, \
                   'conflict kwargs for color'

        color=kwargs.pop('color')
        if color is None: # use color cycle by default
            color=get_next_color_in_cycle(ax)

        kwargs['colors']=[color]

    ## linestyle, linewidth
    for listks in [['linewidth', 'lw'], ['linestyle', 'ls']]:
        kname=listks[0]

        ks=[k for k in listks if k in kwargs]
        if ks:
            assert len(ks)>1 or f'{kname}s' not in kwargs, \
                   f'conflict kwargs for {kname}s'
            lv=kwargs.pop(ks[0])
            kwargs[kname+'s']=lv

    # label
    label=kwargs.pop('label', None)

    # plot contour
    contours=ax.contour(xcents, ycents, dens, levels=vlevels, **kwargs)

    # label
    if label is not None:
        c=contours.collections[0]
        c.set_label(label)
        update_handler_for_contour(c, fill=False)

    return contours

def plot_2d_scatter(ax, xs, ys, s=5, **kwargs):
    '''
        scatter plot for 2d data
    '''
    return ax.scatter(xs, ys, s=s, **kwargs)

### mixed plotting fashion

def plot_2d_joint(axs, xs, ys,
                    xlim=None, ylim=None, hlim='eq0',
                    params_2d=None, params_hist=None):
    '''
        joint plot of 2d data

        Parameters:
            hlim: None, str 'equal', 'eq', 'eq0', or pair of float
                limit of histogram

                if None, do nothing

                'equal', 'eq', 'eq0': equal limit for hist of xs, ys
                    'eq0' to set 0 for lower limit

                pair of float, limit [h0, h1]
    '''
    xs=np.asanyarray(xs)
    ys=np.asanyarray(ys)

    ax, axx, axy=axs  # 3 axes given

    # filter by lim
    xs, ys=filter_by_lim(xs, ys, xlim=xlim)
    ys, xs=filter_by_lim(ys, xs, xlim=ylim)

    # plot 2d
    if params_2d is None:
        params_2d=dict(s=1)
    ax.scatter(xs, ys, **params_2d)

    # hist plot for xs/ys
    if params_hist is None:
        params_hist={}
    plot_hist(axx, xs, **params_hist)
    plot_hist(axy, ys, orientation='horizontal', **params_hist)

    # set ylim of axx and xlim of axy equal
    if hlim is not None:
        if isinstance(hlim, str):
            assert hlim in ['equal', 'eq', 'eq0']

            xh0, xh1=axx.get_ylim()
            yh0, yh1=axy.get_xlim()

            h0, h1=min(xh0, yh0), max(xh1, yh1)
            if hlim=='eq0':
                h0=0

            hlim=[h0, h1]

        axx.set_ylim([h0, h1])
        axy.set_xlim([h0, h1])

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

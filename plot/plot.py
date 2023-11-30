#!/usr/bin/env python3

'''
    simple and useful functions to plot
'''

import numpy as np
import numbers

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ._tools_plot import filter_by_lim, calc_density_map_2d, quants_to_levels
from .colors import get_next_color_in_cycle
from .legend import handler_nonfill, update_handler_for_contour
from .transforms import (CompositeAxisScaleTrans,
                         get_transforms_sr, combine_paths_transforms)
from ._tools_class import bind_new_func_to_instance_by_trans

__all__=['plot_hist', 'plot_cumul',
         'plot_2d_hist', 'plot_2d_contour', 'plot_2d_scatter',
         'plot_2d_bins',
         'plot_2d_joint']

# distribution plot

## 1d
def plot_hist(ax, xs, bins=50, xlim=None, density=True, histtype='step', **kwargs):
    '''
        histogram plot

        hujh-friendly kwargs
    '''
    return ax.hist(xs, bins=bins, density=density, histtype=histtype,
                range=xlim, **kwargs)

def plot_cumul(ax, xs, xlim=None, norm=True, **kwargs):
    '''
        cumulative plot
    '''
    xs=np.asanyarray(xs)

    # limit cut
    xs=filter_by_lim(xs, xlim=xlim)

    # sort
    n=len(xs)
    fracs=np.arange(n+1)
    if norm:
        fracs=fracs/n
    
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
                    levels=None,
                    fill=False,
                    show_points=False, remove_covered=True, kws_points={'s': 1},
                    label_points=False,
                    **kwargs):
    '''
        contour plot

        use `contourf` if fill is bool True, and args `cmap` or `colors` given

        Paramters:
            args: (xs, ys) or (dens, xcents, ycents)
                data to plot

            kde: bool
                whether use KDE to compute density map

            bins, xlim, ylim, limtype: kwargs to calculate density map

            levels: None, int, or array of numbers
                quantiles of contours to plot
                    fraction of points outside of countour

                if None: use levels=10
                if int n: use `[1/n, 2/n, ..., (n-1)/n]`

            fill: bool, or color (str, array)
                whether to fill the contour

                if only True, treatment depend on `kwargs`
                    if not `colors` or `cmap` given,
                        use 'white' and `plt.contour` function
                    otherwise, use `plt.contourf`

            show_points, remove_covered, kws_points: args to plot scatter of points
                other two works only `show_points` is True
                and `show_points` works only if `xs`, `ys` given

                :param remove_covered:
                    remove points covered by contour

            label_points: bool, default False
                if True, label (if given) would added to scatter plot
                otherwise to contour

        Optional Parameters:
            color(s), linewidth(s) (or lw), linestyle(s) (or ls):
                setup for line of contours
                
                if singular keyword given, use same setup for all contours

                if not `color`(s) given, use color cycle

            label: str
                label for contour or scatter plot of points

                if `show_points` and `label_points`,
                    label would be added to scatter plot
                otherwise to contour
    '''
    isdata_xys=False
    if len(args)==2:
        isdata_xys=True
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
        levels=np.linspace(0, 1, levels+1)[1:-1]
    else:
        assert all([0<=l<=1 for l in levels])
        levels=np.sort(levels)

    # setup of contour's lines
    samestyle=True

    ## color
    ckeys_all=['color', 'colors', 'cmap']  # keys to set color
    ckeys=[k for k in ckeys_all if k in kwargs]
    if len(ckeys)>=2:
        raise ValueError(f'conflict kwargs for color: {ckeys}')
    elif len(ckeys)==0:
        kwargs['color']=None  # default color
        ckey='color'  # set `color` args actually
    else:
        ckey=ckeys[0]

    if ckey=='color':
        color=kwargs.pop('color')
        if color is None: # use color cycle by default
            color=get_next_color_in_cycle(ax)

        kwargs['colors']=[color]
    else:
        samestyle=False

    ### contour or contourf
    if (type(fill) is bool and fill) and  ckey!='color':
        use_contourf=True
        func=ax.contourf
    else:
        use_contourf=False
        func=ax.contour

    ## linestyle, linewidth
    for listks in [['linewidth', 'lw'], ['linestyle', 'ls']]:
        kname=listks[0]

        ks=[k for k in listks if k in kwargs]
        ks_multi=[k for k in listks if k+'s' in kwargs]
        ks_all=ks+ks_multi
        if len(ks_all)>=2:
            raise ValueError(f'conflict kwargs for {kname}s: {ks_all}')
        elif ks_multi:
            if kname=='linestyle':  # ignore different linewidth
                samestyle=False
        elif ks:
            lv=kwargs.pop(ks[0])
            kwargs[kname+'s']=lv

    # label
    label=kwargs.pop('label', None)

    # levels in density map
    if use_contourf and levels[-1]!=1.:
        levels=[*levels, 1.]

    vlevels=quants_to_levels(dens, levels)

    # plot contour
    contours=func(xcents, ycents, dens, levels=vlevels, **kwargs)

    ## fill
    if not use_contourf:
        if type(fill) is bool:
            if fill:
                contours.collections[0].set_fc('white')
        else:
            if isinstance(fill, str) or len(fill) in [3, 4]:
                raise ValueError(
                    f'only bool or color str/array for `fill`, but got: {fill}')
            contours.collections[0].set_fc(fill)

    # plot points scatter
    sca=None
    if isdata_xys and show_points:
        if remove_covered:
            xys=np.column_stack([xs, ys])

            pathcol=contours.collections[0]
            for path in pathcol.get_paths():
                m=path.contains_points(xys)
                xys=xys[~m]

            xs, ys=xys.T

        if 'color' not in kws_points and ckey=='color':
            kws_points['color']=color

        if len(xs)>0:
            sca=ax.scatter(xs, ys, **kws_points)

    # label
    if label is not None:
        if label_points and sca is not None:
            sca.set_label(label)
        else:
            collections=contours.collections
            if not use_contourf:
                if samestyle:   # same style for all contour lines
                    c=collections[0]
                    c.set_label(label)
                    update_handler_for_contour(c, fill=False)
                else:
                    for c, l in zip(collections, levels):
                        s=f'{label}: {l}'
                        c.set_label(s)
                        update_handler_for_contour(c, fill=False)
            else:
                lowers=levels[:-1]
                uppers=levels[1:]
                for c, l, u in zip(collections, lowers, uppers):
                    s=f'{label}: ({l}, {u}]'
                    c.set_label(s)
                    update_handler_for_contour(c, fill=True)

    return contours

def plot_2d_scatter(ax, xs, ys, s=None, random_choice=None,
                        is_semisize=False, unit='points',
                        angles=None, ratios=None, base_axis='y', **kwargs):
    '''
        scatter plot for 2d data

        Parameters:
            random_choice: None, or int
                randomly choose a small subgroup to plot
                    in order to decrease file size

            is_semisize: bool, default False
                whether size given in `s` is for semisize
                    that is, like for circle, radius

                that in initial `ax.scatter` is full size
                    that is diameter for circle

            unit: str, default 'points'
                unit for size
                    s is respective to `unit**2`

                valid str: 'points', 'inches', 'dots', 'x', 'y', 'width', 'height'
                    'x', 'y': x or y in data unit
                    'width', 'height': in axes fraction unit
                    'dots', 'inches': pixels or inches
                    'points': points (1/72 inches)

            angles, ratios: None or array-like
            base_axis: 'x' 'y', default 'y'
                parameters to define additional deformation
                    
                deformation only contains scaling, rotation
                    no shearing
                and for scaling, area of path is kept
                    that means only scaling (a, 1/a)

                ratios: ratio of another axis to base axis

                angles=0 if None
                ratios=1 if None

        other parameters: see `ax.scatter`
    '''
    n=len(xs)
    assert len(ys)==n
    if random_choice is not None:
        inds=np.arange(n)
        inds=np.random.choice(inds, size=random_choice, replace=False)

        xs=[xs[i] for i in inds]
        ys=[ys[i] for i in inds]

        if 'c' in kwargs and not isinstance(kwargs['c'], str):
            c=kwargs['c']
            kwargs['c']=[c[i] for i in inds]

        if s is not None and not isinstance(s, numbers.Number):
            s=[s[i] for i in inds]

    # initial plot
    collection=ax.scatter(xs, ys, s=None, **kwargs)

    # set transform
    assert unit in ['points', 'inches', 'dots', 'x', 'y', 'width', 'height']
    axis='x'
    if unit in ['x', 'y']:
        ptrans=ax.transData
        axis=unit
    elif unit in ['width', 'height']:
        ptrans=ax.transAxes
        axis=dict(w='x', h='y')[unit[0]]

    k=2 if is_semisize else 1
    ltrans=[]  # list of trans
    if unit!='points':
        k*=72  # points = inches/72

        if unit!='inches':
            ltrans.append(ax.figure.dpi_scale_trans)
            func=lambda s0, k=k: k/s0

            if unit!='dots':
                ltrans.append(ptrans)
                func=lambda s0, s1, k=k: s1*k/s0
    if not ltrans:
        func=lambda k=k: k  # func for points, inches
    trans=CompositeAxisScaleTrans(ltrans, func, axis=axis)

    collection.set_transform(trans)

    # set sizes
    if s is None:
        s=collection.get_sizes()/trans.get_scale()**2
    elif isinstance(s, numbers.Number):
            s=[s]
    collection.set_sizes(s)

    # deformation
    if angles is not None or ratios is not None:
        angles=0 if None else angles
        ratios=1 if None else ratios
        trans_sr=get_transforms_sr(angles=angles, ratios=ratios, base_axis=base_axis)

        func=lambda res, t1=trans_sr: combine_paths_transforms(res, t1)
        bind_new_func_to_instance_by_trans(collection, 'get_transforms', func)

    ax.autoscale_view()

    return collection

def plot_2d_bins(ax, xs, ys, bins=5, binlim=None, binxoy='x',
                    fagg='median', ferr=None,
                    binagg='binmid', binerr=None, marker='o', **kwargs):
    '''
        plot aggregation in bins

        Parameters:
            bins, binlim: args for bin split
                bins: same as `pandas.cut`
                    int, list of scalars or IntervalIndex

                binlim: [b0, b1] in which b0, b1 float or None
                    if None, use min or max as default

                    it works only when `bins` is given in int

            binxoy: 'x' or 'y'
                bin data at x or y data

            fagg, ferr: args for aggregation in another data
                fagg: func, str, e.g. 'mean', 'median'

                ferr: None, func, or str e.g. 'std', 'sem'
                    'sem': standard error of mean

                    if None, not plot err

            binagg, binerr: args for aggregation in bin data
                binagg: 'binmid' or args as `fagg`
                    if 'binmid', just use middle of bin

                binerr: err agg
                    it works only when `binagg` != 'binmid'

            marker, kwargs: arguments for plot
    '''
    assert binxoy in ['x', 'y']
    if binxoy=='y':
        xs, ys=ys, xs

    # split bins
    if isinstance(bins, numbers.Number) and binlim is not None:
        x0, x1=binlim
        if x0 is None:
            x0=np.min(xs)
        if x1 is None:
            x1=np.max(xs)

        bins=np.linspace(x0, x1, bins+1)

    categories=pd.cut(xs, bins=bins, ordered=True, retbins=False)

    # group and aggregate
    aggfuncs=dict(yagg=('y', fagg), cnt=('y', 'count'))
    if ferr is not None:
        aggfuncs.update(yerr=('y', ferr))

    if binagg=='binmid':
        df=pd.concat([ys, categories], axis=1, keys='y bin'.split())
    else:
        df=pd.concat([xs, ys, categories], axis=1, keys='x y bin'.split())

        aggfuncs.update(xagg=('x', binagg))
        if binerr is not None:
            aggfuncs.update(xerr=('x', xerr))

    df=df.dropna()

    dataplt=df.groupby('bin').agg(**aggfuncs)
    dataplt=dataplt[dataplt['cnt']>5]
    dataplt.dropna()

    # prepare data to plot
    kws_plt={}
    sx, sy='x', 'y'
    if binxoy=='y':
        sx, sy=sy, sx

    kws_plt[sy]=dataplt['yagg']
    if ferr is not None:
        kws_plt[sy+'err']=dataplt['yerr']

    if binagg=='binmid':
        kws_plt[sx]=[interval.mid for interval in dataplt.index]
    else:
        kws_plt[sx]=dataplt['xagg']
        if binerr is not None:
            kws_plt[sx+'err']=dataplt['yerr']

    return ax.errorbar(**kws_plt, marker=marker, **kwargs)

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

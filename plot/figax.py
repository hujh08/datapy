#!/usr/bin/env python3

'''
    module to create frequently-used layout of axes
'''

import numbers

import numpy as np
import matplotlib.pyplot as plt

from .layout import RectManager, Rect
from ._tools_layout import (is_fig_obj, is_axes_obj)

__all__ = ['get_figaxes_grid', 'get_figaxes_in_axes',
           'get_figaxes_joint', 'get_figaxes_residual']

# axes in grid
def get_figaxes_grid(nrows=1, ncols=1,
                        loc=[0.1, 0.8], locing=None, locunits=None,
                        at=None, figsize=None,
                        rects='row', origin_upper=False,
                        sharex=False, sharey=False,
                        return_mat=True, squeeze=True,
                        wspaces=0.01, hspaces=0.01,
                        wpunits=None, hpunits=None,
                        ratios_w=1, ratios_h=1, ratio_wh=None,
                        minw=None, minh=None,
                        unit_minw=None, unit_minh=None, style='tight grid'):
    '''
        axes in grid

        =========
        To locate axes in grid, it is enough to specify
            loc: relative to whole axes in parent rect
            ratios: of width/height among axes
            ratios: between w and h of one ax, aka aspect
            wspaces/hspaces: between nearby axes
        among which
            - ratios of w/h and aspect are dimensionless value

            - loc and w/hspaces are some kind of distances
                besides the value, they must have some unit
                    like w/h of figure or base axes by default

        ========
        Parameters:
            nrows, ncols: int
                number of rows/cols in grid
                ny, nx

            loc, locing, locunits, at: args to set location at parent
                specify location [x0, y0, x1, y1]
                    see `RectGrid.set_loc` for detail

                loc, locunits: params of the location
                    (xl0, yl0, xl1, yl1)
                    see later description for more detail
                locing: how the params act

                3 types for loc, locunits:
                    scalar: xyl
                        xl0=yl0=xl1=yl1=xyl
                        only scalar `loc` when `locing='margin'`
                    2-tuple: (xyl0, xyl1)
                        (xl0, xl1)=(xl1, yl1)=(xyl0, xyl1)
                    4-tuple: (xl0, yl0, xl1, yl1)

                locing: None, 'wh', 'xy', 'margin', 'm',
                            or 2-str of 'xwm', default: None
                    default None:
                        depends on whether scalar given for `loc`
                            'm'  if both `loc`, 'locunits' scalar
                                    or `locunits` is 'ticksep'
                            'wh' otherwise

                    'm': same as 'margin'

                    2-str of 'xwm':
                        e.g. 'xx', 'ww', 'mm'

                `locunits`: list of dist units
                    3 type supported for unit: str, int, or `LnComb`-like
                        LnComb-like: object with `to_lncomb`
                        str: 'figure', 'grid', ...
                             'inches', 'pixel', 'points', ...
                             'ticksep[,k[,k=v]]', 'ticksep*,[k=v]'
                        int: ith rect in grid (0th rect by `origin_upper`)

                    special unit: ticksep
                        func type unit
                            default:
                                ticksep(ntick=1, nlab=1, out=False,
                                            lab=True, pad=True)

                        examples of pair (val, unit):
                            (3, 'ticksep'):
                                            ticksep(ntick=3)
                            ((1, 2), 'ticksep'):
                                            ticksep(ntick=1, nlab=2)
                            (2, 'ticksep,nlab'):
                                            ticksep(nlab=2)
                            (3, 'ticksep*, lab=false':
                                            ticksep(ntick=3, lab=False)
                            (2, 'ticksep,nlab,pad=false'):
                                            ticksep(nlab=2, pad=True)

                    see `RectGrid.grid_dist_by_unit` for detail

            at: None, `plt.figure`, `plt.Axes`, `Rect` or list of `plt.Axes`
                where to create the axes

            figsize: None, or (float, float)
                figure size (w, h) in inches

                acts as upper bound in a loosing way
                    ignored if
                        fig given in `at`, or
                        both w, h determined from constraints, or
                        conflict to current inequality

                if None, use `plt.rcParams['figure.figsize']` 

            rects: str {'all', 'row', 'col'}, or collections of int
                specify in which rects to create axes

                axes returned would have same nested structure

            origin_upper: bool, default False
                whether the index is given with origin in upper

                if True, order of axes starts from upper-left corner

            sharex, sharey: bool, or str, or list of array of int
                specify rects to share x/y-axis

                if str,
                    'all', 'row', 'col'

            return_mat: bool, default: True
                whether to return result as a matrix, with type np.ndarray

                it works only when `rects` in {'row', 'col'}

            squeeze: bool, default: True
                whether to squeeze array if `nrows==1` or `ncols==1`

                it works only when `return_mat` works and True

            wspaces, wpunits: kwargs to specify wspaces
            hspaces, hpunits: kwargs to specify hspaces
                value and unit for distance of wspaces/hspaces
                    
                unit scalar:
                    str: 'figure', 'prect', 'grid', 'rect', or
                         'points', 'pixel', ... or
                         'ticksep'
                    int: ith rect in grid (count from bottom-left)

                see `RectGrid.set_seps` for detail

            ratios_w, ratios_h: None, float, array of float
                ratios of width/height of rects in grid relative to origin rect

                if None, not set

                if float, it means ratios of other rects to origin rect

                if array, its len should be `nx-1`, or `nx` (`ny-1`, `ny`)
                    for `nx-1`, it means `[1, *ratios]`

            ratio_wh: None, float, or 2-tuple
                ratio w/h for one axes or whole axes region
                    (if given (None, int) by tuple)

                if tuple,
                    (int, float), ((int, int), float), (None, int)

                if None, not set
                if float, set for 0th rect
                if (i, r) or ((i, j), r), set for rect in index i or (i, j)
                if (None, r), set for whole rect

            minw, minh: None, float, or 2-tuple
                min width/height for some axes

                if tuple,
                    (int, float), ((int, int), float), (None, int)

                similar as `ratio_wh`

            unit_minw, unit_minh: None or str
                unit of minw, minh
                    None for 'inches'

                same as scalar in `wpunits`, `hpunits`
    '''
    # create rects
    if isinstance(at, Rect):
        manager=at.get_manager()
        grid=at.add_grid(nx=ncols, ny=nrows)
    else:
        manager=RectManager()

        kws1=dict(nx=ncols, ny=nrows)
        if at is None or is_fig_obj(at):
            grid=manager.add_grid(**kws1)
            if is_fig_obj(at):
                manager.set_figure(at)
        elif is_axes_obj(at):
            grid=manager.add_grid_in_axes(at, **kws1)
        else:  # multiply axes
            at=tuple(at)
            if not all(map(is_axes_obj, at)):
                s='got unexpected type in `at`'
                raise TypeError(s)
            grid=manager.add_grid_by_covering_axes(*at, **kws1)

    # constraints
    grid.set_grid_dists(loc=loc, locing=locing, locunits=locunits,
                            origin_upper=origin_upper,
                            wspaces=wspaces, wpunits=wpunits,
                            hspaces=hspaces, hpunits=hpunits,
                            ratios_w=ratios_w, ratios_h=ratios_h,
                            ratio_wh=ratio_wh,
                            minw=minw, minh=minh,
                            unit_minw=unit_minw, unit_minh=unit_minh)

    # create axes
    if at is None and figsize is not None:
        manager.create_figure(figsize=figsize)

    return grid.create_axes(rects=rects, origin_upper=origin_upper,
                                sharex=sharex, sharey=sharey,
                                return_mat=return_mat, squeeze=squeeze,
                                style=style)

## create subplots in existed axes
def get_figaxes_in_axes(axes, nrows=1, ncols=1, loc=[0.1, 0.8], locing='wh',
                            replace=False, **kwargs):
    '''
        create subplots in existed axis

        Parameters:
            loc, locing: location in axes
                default
                    loc: [0.1, 0.8]
                    locing: 'wh'

            replace: bool, default False
                whether to remove the old given `axes`

            other optional kwargs:
                used for ratios in grid
                same as `get_figaxes_grid`

                some args default:
                    ratios_w=1,
                    ratios_h=None,
                    wspaces=0.01,
                    hspaces=None,
                    ratio_wh=None
    '''
    # create axes
    _, axes1=get_figaxes_grid(nrows=nrows, ncols=ncols, loc=loc, locing=locing,
                                at=axes, **kwargs)

    if replace:
        axes.remove()

    return axes1

## frequently-used style of axes

### for joint plot
def get_figaxes_joint(loc=[0.1, 0.8], locing='wh', at=None,
                            ratio_w=0.1, ratio_h=None, ratio_wh=None,
                            wspace=0.01, hspace=None,
                            **kwargs):
    '''
        axes for joint plot of 2d data
            return fig, [ax, axx, axy]
                ax: axes for 2d
                axx, axy: for 1d plot

        Parameters:
            loc, locing: args to determine location in axes

            ratio_w, ratio_h: float, or None for `ratio_h`
                ratio of width/height for 1d x/y plot to 2d axes

                if `ratio_h` is None, use `ratio_w`

            ratio_wh: None or float
                ratio w/h for 2d axes

            wspace, hspace: float, or None for `hspace`
                ratio of width/height of seps to 2d axes

                if `hspace` is None, use `wspace`

        optional kwargs:
            pass to `get_figaxes_grid` except
                sharex, sharey, origin_upper
                    (fix to 'col', 'row', False)
    '''
    if ratio_h is None:
        ratio_h=ratio_w

    if hspace is None:
        hspace=wspace

    i2d, ix, iy=0, 2, 1  # index for 2d, x, y axes

    if ratio_wh is not None:
        assert isinstance(ratio_wh, numbers.Number)
        ratio_wh=(i2d, ratio_wh)  # aspect for 2d axes

    # create fig, axes
    fig, (ax, axx, axy)=get_figaxes_grid(2, 2, loc=loc, locing=locing,
                            at=at, origin_upper=False,
                            ratios_w=ratio_w, ratios_h=ratio_h,
                            ratio_wh=ratio_wh,
                            wspaces=wspace, hspaces=hspace,
                            rects=[i2d, ix, iy],
                            sharex='col', sharey='row', **kwargs)

    # tick params of axx/axy
    axx.tick_params(labelleft=False, left=False, right=False, top=False)
    axy.tick_params(labelbottom=False, bottom=False, top=False, right=False)

    return fig, (ax, axx, axy)

### for residual plot
def get_figaxes_residual(ncols=1, loc=[0.1, 0.8], locing='wh', at=None,
                                sharex='col', sharey=False,
                                ratios_w=1, ratio_h=0.5, ratio_wh=None,
                                wspaces=0.01, hspace=0.01,
                                **kwargs):
    '''
        axes for residual plot of 2d data
            return [[ax, axr]] (list of [ax, axr], each for pair of variables)
                ax: axes for 2d
                axr: for residual plot

        Parameters:
            ncols: int
                num of columns to create

                `nrows` is fixed to 2

            loc, locing: args to determine location in axes

            ratios_w: float, or array of float
                ratios of widths relative to that of origin 2d axes

            ratio_h: float
                ratio of height of `axr` to that of `ax`

            ratio_wh: None or float
                ratio w/h for 2d axes `ax`

            wspaces, hspace: same as `ratios_w`, `ratio_h`
                ratio of width/height of seps to origin 2d axes
    '''
    fig, axes=get_figaxes_grid(nrows=2, ncols=ncols, loc=loc, locing=locing,
                            at=at, origin_upper=True,
                            ratios_w=ratios_w, ratios_h=[1, ratio_h],
                            ratio_wh=ratio_wh,
                            wspaces=wspaces, hspaces=hspace,
                            rects='col',
                            sharex=sharex, sharey=sharey, **kwargs)

    return fig, axes

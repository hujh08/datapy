#!/usr/bin/env python3

'''
    module to create frequently-used layout of axes
'''

import numbers

import numpy as np
import matplotlib.pyplot as plt

from .layout import RectManager, Rect

__all__ = ['get_figaxes_grid', 'get_figaxes_in_axes',
           'get_figaxes_joint', 'get_figaxes_residual']

# axes in grid
def get_figaxes_grid(nrows=1, ncols=1,
                        loc=[0.1, 0.8], locing='wh', locunits=None,
                        at=None, figsize=None,
                        rects='row', origin_upper=False,
                        sharex=False, sharey=False,
                        return_mat=True, squeeze=True,
                        ratios_w=1, ratios_h=1,
                        ratios_wspace=0.01, ratios_hspace=0.01,
                        ratio_wh=None, style='tight grid'):
    '''
        axes in grid

        Parameters:
            nrows, ncols: int
                number of rows/cols in grid
                ny, nx

            
            loc, locing, locunits, at: args to set location at parent
                see `RectGrid.set_loc` for detail

            at: None, `plt.figure`, `plt.Axes`, `Rect`
                where to create the axes

            figsize: None, or (float, float)
                figure size (w, h) in inches

                it works only when fig not exists in `at` and
                    w, h not determined from constraints

                if ratio w/h could be determined,
                    use as large size as possible to keep the ratio

                if None, use `plt.rcParams['figure.figsize']` 

            rects: str {'all', 'row', 'col'}, or collections of int
                specify in which rects to create axes

                axes returned would be same nested structure

            origin_upper: bool, default False
                whether the index is given with origin in upper

                if True, order of axes starts from upper-left corner

            sharex, sharey: bool, or str {'all', 'row', 'col'}, or list of array of int
                specify rects to share x/y-axis

            return_mat: bool, default: True
                whether to return result as a matrix, with type np.ndarray

                it works only when `rects` in {'row', 'col'}

            squeeze: bool, default: True
                whether to squeeze array if `nrows==1` or `ncols==1`

                it works only when `return_mat` works and True

            ratios_w, ratios_h: None, float, array of float
                ratios of width/height of rects in grid relative to origin rect

                if None, not set

                if float, it means ratios of other rects to origin rect

                if array, its len should be `nx-1`, or `nx` (`ny-1`, `ny` respectively)
                    for `nx-1`, it means `[1, *ratios]`

            ratios_wspace, ratios_hsapace: None, float, array of float
                ratios of wspace, hspace relative to origin rect
                Similar as `ratios_w`, `ratios_h`
                    Except, if array, its len must be `nx-1`, `ny-1` respectively

                if None, not set

            ratio_wh: None, float, or tuple (int, float), ((int, int), float), (None, int)
                ratio w/h for one axes or whole axes region (if given (None, int))

                if None, not set
                if float, set for 0th rect
                if (i, r) or ((i, j), r), set for rect in index i or (i, j)
                if (None, r), set for whole rect
    '''
    # create rects
    if isinstance(at, Rect):
        manager=at.get_manager()
        grid=at.add_grid(nx=ncols, ny=nrows)
    else:
        manager=RectManager()

        if at is None or isinstance(at, plt.Figure):
            grid=manager.add_grid(nx=ncols, ny=nrows)
            if isinstance(at, plt.Figure):
                manager.set_figure(at)
        elif isinstance(at, plt.Axes):
            grid=manager.add_grid_in_axes(at, nx=ncols, ny=nrows)
        else:
            raise TypeError(f'unexpected type for `at`: {type(at)}')

    # constraints
    grid.set_grid_ratios(loc=loc, locing=locing, locunits=locunits,
                            origin_upper=origin_upper,
                            ratios_w=ratios_w, ratios_h=ratios_h,
                            ratios_wspace=ratios_wspace, ratios_hspace=ratios_hspace,
                            ratio_wh=ratio_wh)

    # create axes
    if at is None and figsize is not None:
        manager.create_figure(figsize=figsize)

    return grid.create_axes(rects=rects, origin_upper=origin_upper,
                                sharex=sharex, sharey=sharey,
                                return_mat=return_mat, squeeze=squeeze, style=style)

## create subplots in existed axes
def get_figaxes_in_axes(axes, nrows=1, ncols=1, loc=[0.1, 0.8], locing='wh',
                            replace=False, **kwargs):
    '''
        create subplots in existed axis

        :param loc, locing: location in axes

        :param replace: bool, default False
            whether to remove the old given `axes`

        other optional kwargs:
            used for ratios in grid
            same as `get_figaxes_grid`

            including (with default):
                ratios_w=1,
                ratios_h=None,
                ratios_wspace=0.01,
                ratios_hspace=None,
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
                            ratio_wspace=0.01, ratio_hspace=None,
                            **kwargs):
    '''
        axes for joint plot of 2d data
            return [ax, axx, axy]
                ax: axes for 2d
                axx, axy: for 1d plot

        Parameters:
            loc, locing: args to determine location in axes

            ratio_w, ratio_h: float, or None for `ratio_h`
                ratio of width/height for 1d x/y plot to 2d axes

                if `ratio_h` is None, use `ratio_w`

            ratio_wh: None or float
                ratio w/h for 2d axes

            ratio_wspace, ratio_hspace: float, or None for `ratio_hspace`
                ratio of width/height of seps to 2d axes

                if `ratio_hspace` is None, use `ratio_wspace`

        optional kwargs:
            pass to `get_figaxes_grid` except
                sharex, sharey, origin_upper
                    (fix to 'col', 'row', False)
    '''
    if ratio_h is None:
        ratio_h=ratio_w

    if ratio_hspace is None:
        ratio_hspace=ratio_wspace

    i2d, ix, iy=0, 2, 1  # index for 2d, x, y axes

    if ratio_wh is not None:
        assert isinstance(ratio_wh, numbers.Number)
        ratio_wh=(i2d, ratio_wh)  # aspect for 2d axes

    # create fig, axes
    fig, (ax, axx, axy)=get_figaxes_grid(2, 2, loc=loc, locing=locing,
                            at=at, origin_upper=False,
                            ratios_w=ratio_w, ratios_h=ratio_h,
                            ratio_wh=ratio_wh,
                            ratios_wspace=ratio_wspace, ratios_hspace=ratio_hspace,
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
                                ratios_wspace=0.01, ratio_hspace=0.01,
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

            ratios_wspace, ratio_hspace: same as `ratios_w`, `ratio_h`
                ratio of width/height of seps to origin 2d axes
    '''
    fig, axes=get_figaxes_grid(nrows=2, ncols=ncols, loc=loc, locing=locing,
                            at=at, origin_upper=True,
                            ratios_w=ratios_w, ratios_h=[1, ratio_h],
                            ratio_wh=ratio_wh,
                            ratios_wspace=ratios_wspace, ratios_hspace=ratio_hspace,
                            rects='col',
                            sharex=sharex, sharey=sharey, **kwargs)

    return fig, axes

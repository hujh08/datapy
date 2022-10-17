#!/usr/bin/env python3

'''
    module to create frequently-used layout of axes
'''

import numbers

import numpy as np

from .layout import RectManager

__all__ = ['get_figaxes_grid', 'get_figaxes_joint', 'get_figaxes_in_axes',
           'get_figaxes_residual']

# axes in grid
def get_figaxes_grid(nrows=1, ncols=1, loc=[0.1, 0.8], figsize=None,
                        rects='row', origin_upper=False,
                        sharex=False, sharey=False,
                        return_mat=True, squeeze=True,
                        ratios_w=1, ratios_h=None,
                        ratios_wspace=0.01, ratios_hspace=None,
                        ratio_wh=None, style='tight grid'):
    '''
        axes in grid

        Parameters:
            nrows, ncols: int
                number of rows/cols in grid
                ny, nx

            loc: [x0, w] or [x0, y0, w, h]
                rectangle of whole axes
                in unit of fraction in figure

                if [x0, w], means y0, h = x0, w

            figsize: None, or (float, float)
                figure size (w, h) in inches

                it works only when fig not created and
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

            ratios_w, ratios_h: float, array of float
                ratios of width/height of rects in grid

                if float, or `len(ratios_w, or ratios_h) == nx-1 (or ny-1)`
                    it means ratios with respect to rect[0, 0] in left-bottom

                if `ratios_h == None`, use `ratios_h = ratios_w`

            ratios_wspace, ratios_hsapace: float, array of float
                ratios of wspace, hspace with respect to origin rect
                similar as `ratios_w`, `ratios_h`

            ratio_wh: None, float, or tuple (int, float), ((int, int), float), (None, int)
                ratio w/h for one axes or whole axes region (if given (None, int))

                if None, not set
                if float, set for 0th rect
                if (i, r) or ((i, j), r), set for rect in index i or (i, j)
                if (None, r), set for whole rect
    '''
    # create rects
    manager=RectManager()
    grid=manager.add_grid(nx=ncols, ny=nrows)

    # constraints
    grid.set_grid_ratios(loc=loc, origin_upper=origin_upper,
                            ratios_w=ratios_w, ratios_h=ratios_h,
                            ratios_wspace=ratios_wspace, ratios_hspace=ratios_hspace,
                            ratio_wh=ratio_wh)

    # create axes
    if figsize is not None:
        manager.create_figure(figsize=figsize)

    return grid.create_axes(rects=rects, origin_upper=origin_upper,
                                sharex=sharex, sharey=sharey,
                                return_mat=return_mat, squeeze=squeeze, style=style)

## for joint plot
def get_figaxes_joint(loc=[0.1, 0.8], ratio_w=0.1, ratio_h=None, ratio_wh=None,
                                      ratio_wspace=0.01, ratio_hspace=None,
                                      **kwargs):
    '''
        axes for joint plot of 2d data
            return [ax, axx, axy]
                ax: axes for 2d
                axx, axy: for 1d plot

        Parameters:
            loc: [x0, w] or [x0, y0, w, h]
                rectangle of whole axes
                in unit of fraction in figure

                if [x0, w], means y0, h = x0, w

            ratio_w, ratio_h: float, or None for ratio_h
                ratio of width/height for 1d x/y plot to 2d axes

                if ratio_h is None, use ratio_w

            ratio_wh: None or float
                ratio w/h for 2d axes
            
            ratio_wspace, ratio_hspace: float, or None for ratio_hspace
                ratio of width/height of seps to 2d axes

        optional kwargs:
            pass to `get_figaxes_grid` except
                sharex, sharey, origin_upper
                    (fix to 'col', 'row', False)
    '''
    if ratio_h is None:
        ratio_h=ratio_w

    if ratio_hspace is None:
        ratio_hspace=ratio_hspace

    i2d, ix, iy=0, 2, 1  # index for 2d, x, y axes

    if ratio_wh is not None:
        assert isinstance(ratio_wh, numbers.Number)
        ratio_wh=(i2d, ratio_wh)  # aspect for 2d axes

    # create fig, axes
    fig, (ax, axx, axy)=get_figaxes_grid(2, 2, loc=loc, origin_upper=False,
                            ratios_w=ratio_w, ratios_h=ratio_h,
                            ratio_wh=ratio_wh,
                            ratios_wspace=ratio_wspace, ratios_hspace=ratio_hspace,
                            rects=[i2d, ix, iy],
                            sharex='col', sharey='row', **kwargs)

    # tick params of axx/axy
    axx.tick_params(labelleft=False, left=False, right=False, top=False)
    axy.tick_params(labelbottom=False, bottom=False, top=False, right=False)

    return fig, (ax, axx, axy)

## for residual plot
def get_figaxes_residual(ncols=1, loc=[0.1, 0.8],
                                ratios_w=1, ratio_h=0.5, ratio_wh=None,
                                ratios_wspace=0.01, ratio_hspace=0.01,
                                sharex='col', sharey='row',
                                **kwargs):
    '''
        axes for residual plot of 2d data
            return [[ax, axr]]
                ax: axes for 2d
                axr: for residual plot

        Parameters:
            loc: [x0, w] or [x0, y0, w, h]
                rectangle of whole axes
                in unit of fraction in figure

                if [x0, w], means y0, h = x0, w

            ratios_w, ratio_h: array of float, float respectively
                ratio of width for to 2d axes

                if ratio_h is None, use ratio_w

            ratio_wh: None or float
                ratio w/h for 2d axes
            
            ratios_wspace, ratio_hspace: array of float, float respectively
                ratio of width/height of seps to 2d axes

        optional kwargs:
            pass to `get_figaxes_grid` except
                sharex, sharey, origin_upper
                    (fix to 'col', 'row', False)
    '''
    fig, axes=get_figaxes_grid(nrows=2, ncols=ncols, loc=loc, origin_upper=True,
                            ratios_w=ratios_w, ratios_h=[1, ratio_h],
                            ratio_wh=ratio_wh,
                            ratios_wspace=ratios_wspace, ratios_hspace=ratio_hspace,
                            rects='col',
                            sharex=sharex, sharey=sharey, **kwargs)

    return fig, axes

## create subplots in existed axes
def get_figaxes_in_axes(axes, nrows=1, ncols=1, loc=[0.1, 0.8], replace=False,
                            rects='row', origin_upper=False,
                            sharex=False, sharey=False,
                            return_mat=True, squeeze=True, style='tight grid',
                            **kwargs):
    '''
        create subplots in existed axis

        :param loc: location in axes

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
    # create rects
    manager=RectManager()
    grid=manager.add_grid_in_axes(axes, nx=ncols, ny=nrows)

    # constraints
    grid.set_grid_ratios(loc=loc, origin_upper=origin_upper, **kwargs)

    # create axes
    _, axes1=grid.create_axes(rects=rects, origin_upper=origin_upper,
                                sharex=sharex, sharey=sharey,
                                return_mat=return_mat, squeeze=squeeze, style=style)

    if replace:
        axes.remove()

    return axes1

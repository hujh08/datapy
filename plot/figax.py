#!/usr/bin/env python3

'''
    module to create frequently-used layout of axes
'''

import numbers

import numpy as np

from .layout import RectManager

__all__ = ['get_figaxes_grid', 'get_figaxes_joint']

# axes in grid

def get_figaxes_grid(nrows=1, ncols=1, figsize=None,
                        sharex=False, sharey=False,
                        return_rects='row', return_mat=True, squeeze=True,
                        loc=[0.1, 0.8],
                        ratios_w=1, ratios_h=None,
                        ratios_wspace=0.01, ratios_hspace=None,
                        mat=True):
    '''
        axes in grid

        Parameters:
            nrows, ncols: int
                number of rows/cols in grid
                ny, nx

            sharex, sharey: bool, or str {'all', 'row', 'col'}, or list of array of int
                specify rects to share x/y-axis

            return_rects: str {'all', 'row', 'col'}, or collections of int
                specify in which rects to create axes

                axes returned would be same nested structure

            return_mat: bool, default: True
                whether to return result as a matrix, with type np.ndarray

                it works only when `return_rects` in {'row', 'col'}

            squeeze: bool, default: True
                whether to squeeze array if `nrows==1` or `ncols==1`

                it works only when `return_mat` works and True

            loc: [x0, w] or [x0, y0, w, h]
                rectangle of whole axes
                in unit of fraction in figure

                if [x0, w], means y0, h = x0, w

            ratios_w, ratios_h: float, array of float
                ratios of width/height of rects in grid

                if float, or `len(ratios_w, or ratios_h) == nx-1 (or ny-1)`
                    it means ratios with respect to rect[0, 0] in left-bottom

                if `ratios_h == None`, use `ratios_h = ratios_w`

            ratios_wspace, ratios_hsapace: float, array of float
                ratios of wspace, hspace with respect to rect[0, 0] in left-bottom
                similar as `ratios_w`, `ratios_h`
    '''
    # create rects
    manager=RectManager()
    grid=manager.add_grid(nx=ncols, ny=nrows)
    rects=grid.get_rects(return_rects)

    # constraints

    ## location of grid
    if len(loc)==2:
        x0, w=y0, h=loc
    else:
        x0, y0, w, h=loc
    grid.set_loc_at([x0, y0, w, h], locing='wh')

    ## ratio of widths/heights with respect to axes[0, 0] at left-bottom
    grid.set_dists_ratio(ratios_w, 'width')

    if ratios_h is None:
        ratios_h=ratios_w
    grid.set_dists_ratio(ratios_h, 'height')

    ## ratio of wspace/hspace with respect to axes[0, 0]
    rect0=grid[0, 0]
    grid.set_seps_ratio_to(rect0.width, ratios_wspace, axis='x')

    if ratios_hspace is None:
        ratios_hspace=ratios_wspace
    grid.set_seps_ratio_to(rect0.height, ratios_hspace, axis='y')

    # sharex, sharey
    kwargs={}
    if sharex:
        if type(sharex) is bool:
            sharex='all'
        kwargs['sharex']=grid.get_rects(sharex)
    if sharey:
        if type(sharey) is bool:
            sharey='all'
        kwargs['sharey']=grid.get_rects(sharey)

    # create axes
    if figsize is not None:
        manager.create_figure(figsize=figsize)
    fig, axes=manager.create_axes_in_rects(rects,
                style='tight grid', **kwargs)

    ## to matrix
    if return_rects in ['row', 'col'] and return_rects:
        axes=np.asarray(axes)
        if squeeze and (nrows==1 or ncols==1):
            axes=np.ravel(axes)
            if len(axes)==1:
                axes=axes[0]

    return fig, axes

## for joint plot

def get_figaxes_joint(rect=[0.1, 0.8], ratio_wh=0.1, ratio_seps=0.01):
    '''
        axes for joint plot of 2d data
            return [ax, axx, axy]
                ax: axes for 2d
                axx, axy: for 1d plot

        Parameters:
            rect: [x0, w] or [x0, y0, w, h]
                rectangle of whole axes
                in unit of fraction in figure

                if [x0, w], means y0, h = x0, w

            ratio_wh: float or pair of float
                ratio of width/height for 1d x/y plot to 2d axes

            ratio_seps: float or pair of float
                ratio of width/height of seps to 2d axes

    '''
    if isinstance(ratio_wh, numbers.Number):
        rw=rh=ratio_wh
    else:
        rw, rh=ratio_wh

    if isinstance(ratio_seps, numbers.Number):
        rwsep=rhsep=ratio_seps
    else:
        rwsep, rhsep=ratio_seps

    # create fig, axes
    fig, (ax, axx, axy)=get_figaxes_grid(2, 2, loc=rect,
                            ratios_w=rw, ratios_h=rh,
                            ratios_wspace=rwsep, ratios_hspace=rhsep,
                            return_rects=[0, 2, 1],
                            sharex='col', sharey='row')

    # tick params of axx/axy
    axx.tick_params(labelleft=False, left=False, right=False, top=False)
    axy.tick_params(labelbottom=False, bottom=False, top=False, right=False)

    return fig, (ax, axx, axy)

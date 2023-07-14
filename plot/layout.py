#!/usr/bin/env python3

'''
    module to handle layout of axes in figure

    Hierarchy of class in this module:
        - RectManager: manager of grids and variables
            hold a LinearManager for variables,
                which are 1d point in grids, i.e. (x0, x1, ..., y0, y1, ...)
            accept register of grid,
                allocating each a unique id,
            root grid,
                standing for whole figure

        - RectGrid: grid to place rectangle
            for grid (nx, ny)， there are nx*ny base rectangles
            An rectangle is created on one or collections of these base regions

        - Rect: rectangle
            final axes would be created in a rectangle

            each rect could be locate by 4 variables
                (x0, x1, y0, y1)
            that is coordinates of its corners

        - Point1d: 1d coordinates in grid
            basic variables to trace and calculate

        - Dist1d: distance between Point1ds
            constraints on dist is frequently used

    unit of constant in linear manager is 'inches'
'''

import numbers
from collections import abc

import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .linear import LinearManager, LnComb
from ._tools_layout import map_to_nested, squeeze_nested, confirm_arg_in
from .size import Units, fontsize_in_pts
from ._tools_class import add_proxy_method
from .params import params_create_axes

class RectManager:
    '''
        manager of rectangles
    '''
    def __init__(self, size=1024, size_vs=None, precise_vs=None):
        '''
            init of manager

            Parameters:
                size: int, or None
                    max num of grid to hold

                    if None, no limit

                size_vs: int or None
                    max num of variables

                    if None, use 16*`size`,
                        averagely 2x2 rect in each grid
                            each rect with 2x2 variables,
                                (x0, x1, y0, y1)

                precise_vs: float or None
                    precise for variables

                    if None, use default setup of `LinearManager`
        '''
        # collection of grid
        self._rectgrids={}
        if size is not None:
            size=int(size)
        self._size=size

        # manger of variables
        if size_vs is None:
            if size is not None:
                size_vs=16*size
        else:
            size_vs=int(size_vs)

        kwargs=dict(size=size_vs) # keywords to create var manager
        if precise_vs is not None:
            kwargs['precise']=precise_vs
        self._vars=LinearManager(**kwargs)

        # root grid/rect for figure
        self._root_grid=RectGrid(1, 1)
        self._root_grid._register_at_manager(self)

        self._root_rect=self._root_grid._get_rect(0, 0)

        ## fix (x0, y0) of root grid to 0
        for k in 'xy':
            p=self._root_grid._get_ith_point(k, 0)
            p._set_to(0)

        # figure instance
        self._fig=None

    # accept register from grid
    def _accept_grid(self, grid):
        '''
            accept register of grid

            return a unique id
        '''
        if self._size is not None:
            assert len(self._rectgrids)<=self._size, \
                    'too many rect grid add, ' \
                    'only support at most %i' % self._size

        name='g%i' % len(self._rectgrids)
        assert name not in self._rectgrids

        self._rectgrids[name]=grid
        return name

    # grid
    def add_grid(self, nx=1, ny=1):
        '''
            add grid in root

            Parameters:
                nx, ny: int
                    ncols and nrows of grid
        '''
        return self._root_rect._add_grid(nx=nx, ny=ny)

    def add_grid_in_axes(self, axes, nx=1, ny=1):
        '''
            add grid to an existed axes

            Parameters:
                axes: `plt.Axes` instance
                    existed axes to add grid in

                nx, ny: int
                    ncols, ncols of grid
        '''
        assert isinstance(axes, plt.Axes),\
               'only support `plt.Axes` to add grid in'

        # figure
        fig=axes.get_figure()
        self.set_figure(fig)

        rect_root=self.add_grid(1, 1).get_rect(0)  # add a layer

        # position of axes
        # (x0, y0), (x1, y1)=axes.transAxes.transform([[0, 0], [1, 1]])
        # (x0, y0), (x1, y1)=fig.transFigure.inverted().transform([[x0, y0], [x1, y1]])
        # unitx, unity=self.width, self.height

        x0, y0, x1, y1=axes.bbox.extents  # position in pixel
        unitx=unity=self.get_size_unit('px')

        rect_root.x0=x0*unitx
        rect_root.y0=y0*unity
        rect_root.x1=x1*unitx
        rect_root.y1=y1*unity

        grid=rect_root.add_grid(nx=nx, ny=ny)

        return grid

    # getter, setter for root rect
    for k_ in 'width height'.split():
        for t_ in 'getter setter'.split():
            add_proxy_method(locals(),
                '%s_%s' % (t_[:3], k_), '_root_rect', t_)
    del k_, t_

    ## properties
    width =property(get_width, set_width)
    height=property(get_height, set_height)

    left  =property(lambda self: self._root_rect.get_left())
    right =property(lambda self: self._root_rect.get_right())
    bottom=property(lambda self: self._root_rect.get_bottom())
    top   =property(lambda self: self._root_rect.get_top())

    rect=property(lambda self: self._root_rect)

    # size unit
    _UNITS=set(Units.keys())
    _UNITS.update(['px', 'pixel'])

    _VAR_INV_DPI='idp'   # var for inverse of 'dpi'

    def get_size_unit(self, unit):
        '''
            return LnComb for a unit

            valid unit:
                inches: default

                cm, mm, ...: length

                points: 1/72 inches
                    always used in fontsize, labelsize, et. al.

                px, pixel: pixel
                    1/dpi 
        '''
        assert unit in self._UNITS, \
            'only support units: %s' % str(self._UNITS)

        # convert alias
        if unit=='px':
            unit='pixel'

        # construct LnComb
        if unit in Units:
            return LnComb.lncomb(Units[unit])

        if unit=='pixel':
            return LnComb([self._VAR_INV_DPI], 1, 0)

    def get_points_size(self, size=1):
        '''
            return LnComb for given `size` in unit of points
            const of LnComb is in unit of inches

            many elements in matplotlib accept size in points
                like fontsize, line width, labelsize

            Parameters:
                size: float
                    in unit of points
        '''
        assert isinstance(size, numbers.Real), \
            'only support float for size'

        return size*self.get_size_unit('points')

    def get_fontsize(self, size=None):
        '''
            return font size in inches
                with LnComb type

            Parameters:
                size: None, float, or str
                    fontsize

                    if float,
                        absolute value of fontsize in unit points

                    if str,
                        'xx-small', 'x-small', 'small', 'medium', 'large', 
                        'x-large', 'xx-large', 'larger', 'smaller'
                    see `matplotlib.font_manager.font_scalings`
                        for details
        '''
        pts=self.get_size_unit('points')  # points size in inches
        size=fontsize_in_pts(size)
        return size*pts

    # tick size
    def get_ticksize(self, axis, item='tick', which='major'):
        '''
            get size of x/y-tick
                for label, pad, tick

            ==============
            Parameters:
                axis: 'x' or 'y'
                    axis
                    only 'x' or 'y'

                item: 'tick', 'lab', 'pad'
                    which item

                which: 'major' or 'minor'
                    not acts for 'lab'

            ==============
            Return:
                LnComb:
                    with const in unit of inches
        '''
        # check
        axes=['x', 'y']
        confirm_arg_in(axis, axes, 'axis')

        items=['tick', 'lab', 'pad']
        confirm_arg_in(item, items, 'item')

        whichs=['major', 'minor']
        confirm_arg_in(which, whichs, 'which')

        # item
        if item=='lab': # item
            v=plt.rcParams[axis+'tick.labelsize']
            return self.get_fontsize(v)

        # tick or pad
        if item=='tick':
            item='size'
        key='%stick.%s.%s' % (axis, which, item)
        v=plt.rcParams[key]
        return self.get_points_size(v)

    def get_sepsize_tick(self, axis, out=True, nchar=1):
        '''
            separation size for whole x/y-tick plot
                along perpendicular direction
            that is sum of sizes of label, tick and pad

            always used to constrain min separation between axes,
                such that not overlap

            ==============
            Parameters:
                axis: 'x' or 'y'

                out: bool
                    whether has tick out

                    tick direction could be 'in', 'out', 'inout'

                nchar: int
                    number of chars in perpendicular direction

                    alway 1 char for x-axis along y-direction

            ==============
            Return:
                LnComb:
                    with const in unit of inches
        '''
        labsize=self.get_ticksize(axis, 'lab').const  # lab for tick lable
        padsize=max([self.get_ticksize(axis, 'pad', t).const
                        for t in ['major', 'minor']])

        v=nchar*labsize+padsize

        if out:
            ticksize=max([self.get_ticksize(axis, 'tick', t).const
                            for t in ['major', 'minor']])
            v+=ticksize

        return LnComb.lncomb(v)

    ## sep size for axis
    def get_ticksepsize_axis(self, axis,
                               nchar_tick=1, nchar_lab=1,
                               tick_out=False, lab=True):
        '''
            min separation size for x/y-axis plot
            that is sum of size of tick plot,
                                   axes label,
                                   axes pad
        '''
        s=self.get_sepsize_tick(axis, out=tick_out, nchar=nchar_tick)

        if lab:
            labsize=self.get_fontsize(plt.rcParams['axes.labelsize'])
            labpad=self.get_points_size(plt.rcParams['axes.labelpad'])

            s=s+nchar_lab*labsize+labpad

        return s

    # constraint for var manager
    def _add_lncomb(self, left, right=0, ineq=False):
        '''
            add constraint as
                left = right

            Parameters:
                left, right: LnComb-support obj
                    LnComb instance, int, or
                    object with attr `to_lncomb`

                ineq: bool, or str
                    whether add ineq

                    for str, only support:
                        upper: 'upper', 'le', '<='
                        lower: 'lower', 'ge', '>='
        '''
        self._vars.add_lncomb(left, right, ineq=ineq)

    def _add_lncomb_ineq(self, left, right=0, upper=True):
        '''
            add ineq
                left <= right if `upper` is True
                left >= right otherwise
        '''
        self._vars.add_lncomb_ineq(left, right, upper=upper)

    ## evaluate
    def _eval_lncomb(self, lncomb):
        '''
            evaluate a LnComb based on variables in manager

            if not determined, return None
        '''
        return self._vars.eval_lncomb(lncomb)

    def _eval_bounds_of_lncomb(self, lncomb):
        '''
            eval bounds of a linear combination
            return pair (l, u)
                None for non-determined in each side
        '''
        return self._vars.eval_bounds_of_lncomb(lncomb)

    def _eval_ratio_of_lncombs(self, t0, t1, allow_kb=False):
        '''
            evaluate ratio of two terms, t0/t1
            if `allow_kb`
                return k, b, satisfying
                    t0 = k*t1 + b

            if not determined, return None
        '''
        return self._vars.eval_ratio_of_lncombs(t0, t1, allow_kb=allow_kb)

    ## user method
    def eval(self, t):
        '''
            evalue a linear combination object
        '''
        return self._eval_lncomb(t)

    def eval_bounds(self, t):
        '''
            eval bounds of linear combination
            return (l, u)
                if one of `l` or `u` is None,
                    means non-determined
        '''
        return self._eval_bounds_of_lncomb(t)

    def eval_ratio(self, t1, t2, **kwargs):
        '''
            eval t1/t2 or k, b in t1=k*t2+b (kwarg `allow_kb`)
        '''
        return self._eval_ratio_of_lncombs(t1, t2, **kwargs)

    def set_equal(self, t0, *args):
        '''
            set all terms equal

            Parameters:
                t0, *args: LnComb-like
                    terms to be set equal
                        at least 2
        '''
        for t in args:
            self._add_lncomb(t0, t)

    def set_ratio_to(self, terms, ratios):
        '''
            set ratio between terms

            Parameters:
                terms: list of LnComb-like
                    linear combination of variables to set

                ratios: float, or list of float
                    ratios of the terms

                    if list, first 1 could be omit
                    if float, like k,
                        treat as [1, k, ..., k]
        '''
        ratios=_parse_ratios(ratios, len(terms))

        t0, k0=terms[0], ratios[0]
        for ti, ki in zip(terms[1:], ratios[1:]):
            self._add_lncomb(t0*ki, ti*k0)

    ## constraints to points
    def align_points(self, p0, p1, *args):
        '''
            aliang serveral points
                like left of some rectangles

            Parameters:
                p0, p1, *args: Point1D or point-like LnComb
                    points to align
        '''
        self.set_equal(p0, p1, *args)

    ## constraints to distances
    def set_dists_ratio(self, dists, ratios):
        '''
            set ratio between distances

            Parameters:
                dists: list of LineSeg1D-like object
                    dists to set

                ratios: float, or list of float
                    ratio between dists
                    see `set_ratio_to` for detail
        '''
        self.set_ratio_to(dists, ratios)

    def set_dists_equal(self, d0, d1, *args):
        '''
            set distances equal

            Parameters:
                d0, d1, *args: LineSeg1D-like object
                    distances to set equal
        '''
        self.set_equal(d0, d1, *args)

    def set_dists_bound(self, d, *args, upper=True):
        '''
            set dists' bound

            :param upper: bool
                if True, set upper bound
        '''
        for di in args:
            self._add_lncomb_ineq(di, d, upper=upper)

    def set_dists_le(self, maxd, *args):
        '''
            set dists less-equal to a value
        '''
        self.set_dists_bound(maxd, *args, upper=True)

    def set_dists_ge(self, mind, *args):
        '''
            set dists greater-equal to a value
        '''
        self.set_dists_bound(mind, *args, upper=False)

    def set_dists_nonneg(self, *args):
        '''
            set all dists non-neg
        '''
        self.set_dists_ge(0, *args)

    def set_dists_lim(self, lim, *args):
        '''
            set limited range for dists

            allown None for a bound
                if None, skip it
        '''
        l, u=lim

        if l is not None:
            self.set_dists_ge(l, *args)

        if u is not None:
            self.set_dists_le(u, *args)

    ## set to min or max bound
    def set_to_bound(self, d, lu):
        '''
            set dist to a bound

            :param lu: int
                0 for lower, 1 for upper
        '''
        b=self.eval_bounds(d)[lu]
        self.set_equal(d, b)

    def set_to_min(self, d):
        '''
            set dist to its lower bound
        '''
        self.set_to_bound(d, 0)

    def set_to_max(self, d):
        '''
            set dist to its upper bound
        '''
        self.set_to_bound(d, 1)

    ### set width or height to bounds
    def set_width_to_min(self):
        '''
            set width to its lower bound
        '''
        self.set_to_min(self.get_width())

    def set_width_to_max(self):
        self.set_to_max(self.get_width())

    def set_height_to_min(self):
        self.set_to_min(self.get_height())

    def set_height_to_max(self):
        self.set_to_max(self.get_height())

    # create figure
    ## figure size
    def eval_figsize(self):
        '''
            evaluate figure size, (w, h)
            if not determined, return None
        '''
        root=self._root_rect

        w=self.eval(root.width)
        h=self.eval(root.height)

        return w, h

    def eval_wh_ratio(self, **kwargs):
        '''
            evaluate w/h, or k, b for w=k*h+b
        '''
        root=self._root_rect
        return self.eval_ratio(root.width, root.height, **kwargs)

    ## dpi
    def set_dpi(self, dpi):
        '''
            set dpi, dots per inches
        '''
        # inverse of dpi: pixel size in inches, or inches per dots
        ipd=self.get_size_unit('pixel')
        self._add_lncomb(ipd, 1/dpi)

    def eval_dpi(self):
        '''
            eval dpi, dots per inch
        '''
        ipd=self.get_size_unit('pixel')
        inv_dpi=self.eval(ipd)

        if inv_dpi is None:
            return None
        return 1/inv_dpi

    def set_figure(self, fig):
        '''
            set an exsited figure to manager
        '''
        assert isinstance(fig, plt.Figure),\
               'only support set to Figure instance'

        if self._fig is not None:  # already created
            if self._fig is fig:
                return
            raise Exception('figure already created')

        # w, h in inches
        w, h=fig.get_figwidth(), fig.get_figheight()

        self.set_width(w)
        self.set_height(h)

        # dpi
        dpi=fig.get_dpi()
        self.set_dpi(dpi)

        # assign attr
        self._fig=fig

    def create_figure(self, figsize=None, dpi=None, **kwargs):
        '''
            create figure at root rect via `plt.figure`
            return created fig

            if existed, just return it

            Parameters:
                figsize: (w, h) in inches or None
                    figure size to create
                    acts only when fig not created and
                                w, h not determined from constraints

                    allow None for w or h to skip setting it

                    if ratio w/h could be determined,
                        use as large size as possible to keep the ratio

                    if None, use `plt.rcParams['figure.figsize']`

                dpi: float
                    dots per inches
                    only acts when not determined from constraints

                    if None, use `plt.rcParams['figure.dpi']`

                kwargs: optional kwargs for `plt.figure`
        '''
        if self._fig is not None:  # already created
            return self._fig

        root=self._root_rect
        w, h=self.eval_figsize()

        if w is None or h is None:
            self._set_figsize(figsize)

            # eval from figsize
            w, h=self.eval_figsize()
            if w is None or h is None:
                raise ValueError(f'`figsize` "{figsize}"" not enough to fix figure: {(w, h)}')

        # dpi
        px=self.eval_dpi()
        if px is None:
            if dpi is None:
                dpi=plt.rcParams['figure.dpi']
            else:
                assert isinstance(dpi, numbers.Real), \
                    'only allow float for `dpi`'

            self.set_dpi(dpi)
            px=self.eval_dpi()

        fig=plt.figure(figsize=(w, h), dpi=px, **kwargs)
        self._fig=fig

        return fig

    def create_axes_in_rects(self, rects,
                    sharex=None, sharey=None,
                    return_fig=True, **kwargs):
        '''
            create axes in collection of rects

            rects could be given in some organization
                like ndarray, nested list
            axes are returned with same organization

            Parameters:
                rects: Rect, or Iterable
                    collection of rect to create axes

                sharex, sharey: list of collections
                    each entry corresponds to a set of rect
                        to share axis with each other

                return_fig: bool
                    if True, return fig, axes
                    otherwise, return axes

                optional kwargs: used in `Rect.create_axes`
        '''
        fig=self.create_figure()

        axes=self._create_axes_recur(rects, **kwargs)

        # share x/y-axis
        self._set_grps_share_axis('x', sharex, ignore_nonexists=True)
        self._set_grps_share_axis('y', sharey, ignore_nonexists=True)

        if return_fig:
            return fig, axes
        return axes

    ## auxiliary functions
    def _set_figsize(self, figsize=None):
        '''
            set figsize

            :param figsize: None, or (w, h)
                allow None for w or h for not skip it

                if figsize is None, use `plt.rcParams['figure.figsize']`

                if both given not None and ratio in manager has been fixed,
                    use as large size as possible to keep the ratio
        '''
        if figsize is None:
            figsize=plt.rcParams['figure.figsize']

        w, h=figsize
        if w is None and h is None:
            return

        root=self._root_rect
        if w is None:
            root.set_height(h)
            return

        if h is None:
            root.set_width(w)
            return

        assert isinstance(w, numbers.Real) and \
               isinstance(h, numbers.Real), \
                'only allow float for `figsize`'

        # ratio w/h eval
        kb=self.eval_wh_ratio(allow_kb=True)
        if kb is None:
            root.set_width(w)
            root.set_height(h)
        else:
            k, b=kb
            if w>k*h+b:
                root.set_height(h)
            else:
                root.set_width(w)

    def _set_grps_share_axis(self, axis, grps=None, ignore_nonexists=True):
        '''
            set axis share for list of groups
        '''
        if grps is None:
            return

        # only one group
        if all([isinstance(a, Rect) for a in grps]):
            return self._set_axes_share_axis(axis, grps, ignore_nonexists=ignore_nonexists)

        # multiple groups
        for grp in grps:
            assert not isinstance(grp, Rect)
            self._set_axes_share_axis(axis, grp, ignore_nonexists=ignore_nonexists)

    def _set_axes_share_axis(self, axis, rects, ignore_nonexists=True):
        '''
            set axis share between group of rects

            in matplotlib, sharex/sharey is implemented by
                `matplotlib.cbook.Grouper`
        '''
        if len(rects)<2:
            return

        rects=squeeze_nested(rects, is_scalar=lambda a: isinstance(a, Rect))
        if ignore_nonexists:  # ignore rect while no axes exists
            while rects and not rects[0].has_axes():
                rects.pop(0)

            if len(rects)<2:
                return

        rect0=rects.pop(0)
        for ri in rects:
            ri.set_axis_share(axis, rect0, ignore_nonexists=ignore_nonexists)

    def _create_axes_recur(self, rects, **kwargs):
        '''
            recusively create axes in rects
        '''
        return map_to_nested(lambda a: a.create_axes(**kwargs), rects,
                      is_scalar=lambda a: isinstance(a, Rect))

    ## properties
    fig=property(lambda self: self._fig)

    # to string
    def vars_info(self):
        '''
            print info of variables
        '''
        self._vars.info()

class RectGrid:
    '''
        class of grid to place rectangles
    '''
    def __init__(self, ny=1, nx=1, parent=None, manager=None):
        '''
            init of grid

            Parameters:
                nx, ny: int
                    cols and rows in grid

                    default (1, 1)

                parent: Rect instance or None
                    parent rectangle where grid locate

                    root grid has no parent rect

                manager: RectManager instance or None
                    manager of rectangles
        '''
        # nrows, ncols
        assert isinstance(nx, numbers.Integral) and \
               isinstance(ny, numbers.Integral), \
                'only allow integral for (nx, ny)'
        assert nx>0 and ny>0, 'only allow positive for (nx, ny)'
        self._nx=nx
        self._ny=ny

        ## indices along x- and y- axis
        self._xindex=np.arange(self._nx)
        self._yindex=np.arange(self._ny)

        # buffer to place created rects: indexed by (x, y, xspan, yspan)
        self._buf_rects={}

        # parent rect
        if parent is not None:
            assert isinstance(parent, Rect), \
                    'only type Rect for parent rect, '\
                    'but got %s' % (type(parent).__name__)
        self._parent=parent

        # grid manager
        self._manager=None
        if manager is not None:
            return self._register_at_manager(manager)

        ## use manager of parent if it exists
        if self._parent is not None:
            # no recursion to parent of parent
            pm=self._parent._grid._manager
            if pm is not None:
                self._register_at_manager(pm)

    # manager
    def _get_manager(self, recur_if_none=True):
        '''
            return manager
            if None, query parent recursively if `recur`=True

            just get, no register done,
                even when find one and current has no manager

            Parameter:
                recur_if_none: bool
                    whether to query parent recursively
                        to find a manager
        '''
        m=self._manager

        # recursion to query parent
        if m is None and recur_if_none:
            g=self
            while g._parent is not None:
                g=g._parent._grid
                if g._manager is not None:
                    m=g._manager
                    break
        return m

    get_manager=_get_manager

    ## register at manager
    def _register_at_manager(self, manager):
        '''
            register at a manager
            raise Error if already register at another one

            get a unique id from manager
                used to compose unique variables

            if parent exists,
                also register it,
                    or check it to be same if it has a manager
            if no parent, reset parent to root grid (if self not root)
        '''
        assert isinstance(manager, RectManager), \
                'unexpected type for manager: ' + \
                type(manager).__name__

        if self._manager is not None:
            assert self._manager is manager, \
                    'already register at another manager'
            return

        self._manager=manager

        # store name returned from manager
        name=manager._accept_grid(self)
        self._name=name

        # manager in parent
        if self._parent is None:
            g0=manager._root_grid
            if g0 is not self:
                self._parent=manager._root_grid._get_rect(0, 0)
            return

        g1=self._parent._grid
        pm=g1._manager
        if pm is None:  # register parent if it not
            g1._register_at_manager(manager)
            return

        ## otherwise, must same manager
        assert pm is manager, \
                'different manager with parent rect'

    def _register_at_parent_manager(self):
        '''
            register at parent's manager recursively
        '''
        # return if already registered
        if self._manager is not None:
            return

        m=self._get_manager(recur_if_none=True)
        assert m is not None, \
                'no manager found recursively in parent'
        return self._register_at_manager(m)

    ## user methods: getter
    def has_manager(self):
        '''
            whether registered at a manager
        '''
        return self._manager is not None

    def get_manager(self):
        '''
            return manager
            not query parent recursively
        '''
        return self._manager

    def has_name(self):
        '''
            whether has a name from manager
        '''
        return hasattr(self, '_name')

    def get_name(self):
        '''
            return the unique name returned by manager
        '''
        if not self.has_name():
            return None

        return self._name

    @property
    def name(self):
        '''
            return name in manager
        '''
        return self.get_name()

    ## user methods: setter of manager
    def register_at(self, manager=None):
        '''
            register at a manager
            if no manager given, try parent's manager recursively

            Parameters:
                manager: None or RectManager
                    if None, try that of parent
        '''
        if manager is not None:
            self._register_at_manager(manager)
            return

        self._register_at_parent_manager()

    # subgrid
    def add_subgrid(self, nx, ny, indx=0, indy=0):
        '''
            add subgrid `(ny, nx)` at loc `(indy, indx)`

            Parameters:
                nx, ny: int
                    num of rows and coumns for subgrid

                indx, indy: int, slice, or other item object
                    specify rect to place grid in
                    see `get_rect` for detail
        '''
        return self._get_rect(indx=indx, indy=indy)\
                   ._add_grid(nx=nx, ny=ny)

    # rect
    def _get_rect(self, indx=0, indy=0, xspan=None, yspan=None, origin_upper=False):
        '''
            return rect with index (indx, indy)

            base function to get rect

            Parameters:
                indx, indy: int, slice or objects as item of ndarray
                    specify location of rectangle

                xspan, yspan: int, optional
                    span of rect in grid
                    if set to not-None value, indx (or indy) must be int
                    otherwise, just use indx/indy to locate rect

                origin_upper: bool, default False
                    whether the index is given with origin in upper

                    if True, order of axes starts from upper-left corner
        '''
        if xspan is None:
            xs=self._xindex[indx]
            x0, x1=np.min(xs), np.max(xs)
            indx, xspan=x0, x1-x0+1
        else:
            assert isinstance(indx, numbers.Integral), \
                'only allow integral `indx` when `xspan` set'

        if yspan is None:
            ys=self._yindex[indy]
            y0, y1=np.min(ys), np.max(ys)
            indy, yspan=y0, y1-y0+1
        else:
            assert isinstance(indx, numbers.Integral), \
                'only allow integral `indy` when `yspan` set'

        assert xspan>=1 and yspan>=1, 'only allow positive for span'

        # standard index
        x0, y0=self._standard_rect_xyind(indx=indx, indy=indy)
        x1, y1=self._standard_rect_xyind(indx=x0+xspan-1, indy=y0+yspan-1)

        xspan, yspan=x1-x0+1, y1-y0+1
        indx=x0
        if origin_upper:    # use origin lower as default
            indy=self._ny-1-y1
        else:
            indy=y0

        # query buffer of existed rects
        k=(indx, indy, xspan, yspan)
        if k in self._buf_rects:
            return self._buf_rects[k]

        # create new rect
        rect=Rect(self, indx=indx, indy=indy,
                          xspan=xspan, yspan=yspan)
        self._buf_rects[k]=rect

        return rect

    def _get_ith_rect(self, i, origin_upper=False):
        '''
            return ith rect

            rects is ordered in row-first way

            :param origin_upper: bool, default False
                whether the index is given with origin in upper

                if True, order of axes starts from upper-left corner
        '''
        indx, indy=self._standard_rect_xyind_of_ith(i)

        return self._get_rect(indx=indx, indy=indy, origin_upper=origin_upper)

    ## index of base rect
    def _get_num_rect_along_axis(self, axis):
        '''
            return num of rects along a axis
        '''
        return getattr(self, '_n'+axis)

    def _standard_rect_xind(self, indx):
        '''
            standardize x-index of (base) rect
        '''
        assert isinstance(indx, numbers.Integral), \
                'only allow integral for rect x-index'
        return self._xindex[indx]

    def _standard_rect_yind(self, indy):
        '''
            standardize y-index of (base) rect
        '''
        assert isinstance(indy, numbers.Integral), \
                'only allow integral for rect y-index'
        return self._yindex[indy]

    def _standard_rect_xyind(self, indx, indy):
        '''
            standardize index of (base) rect
        '''
        return self._standard_rect_xind(indx),\
               self._standard_rect_yind(indy)

    def _standard_rect_xyind_of_ith(self, i):
        '''
            return standard index for ith rect
                (indx, indy)
        '''
        assert isinstance(i, numbers.Integral), \
            'only support int for order'

        nx, ny=self._nx, self._ny
        n=nx*ny
        if i<0:
            i+=n
        assert 0<=i<n, 'index of rect is out of bounds'

        indy=i//nx
        indx=i-indy*nx
        return indx, indy

    def _standard_rect_xyind_arg(self, *args, return_order=False, **kwargs):
        '''
            stardardize arg of index

            if `return_order`, return int order of rect
        '''
        if len(args)==1 and not kwargs:
            ix, iy=self._standard_rect_xyind_of_ith(args[0])
        else:
            ix, iy=self._standard_rect_xyind(*args, **kwargs)

        if return_order:
            return ix+iy*self._nx
        return ix, iy

    ## user methods: getter
    def get_parent(self):
        '''
            return parent rect
        '''
        return self._parent
    parent=property(get_parent)

    def get_rect(self, i, origin_upper=False):
        '''
            get a rect in Grid

            :param i: int, or tuple (int, int)
                index of rect ot return
                always create new rect

                if int, just return ith rect
                    rects is ordered in row-first way

                if (i, j), means rect in ith row and jth col

            :param origin_upper: bool, default False
                whether the index is given with origin in upper

                if True, order of axes starts from upper-left corner
        '''
        if isinstance(i, numbers.Number):
            return self._get_ith_rect(i, origin_upper=origin_upper)

        iy, ix=i
        return self._get_rect(indx=ix, indy=iy, origin_upper=origin_upper)

    def get_rects(self, arg='row', reverse=False, origin_upper=False):
        '''
            return collection of rects

            Parameters:
                arg: str, or nested list of int or tuple
                    if str, only 'row', 'col', 'all'
                        return all rects organized as
                            'row': 2d array, row-first
                            'col': 2d array, columns-first
                            'all': 1d array, row-first

                    if nested list, scalar is index passed to `__getitem__`
                        e.g. int for ith rect
                    return collection with same organization

                reverse: bool
                    whether reverse order of rects

                    only acts when arg is str

                origin_upper: bool, default False
                    whether the index is given with origin in upper

                    if True, order of axes starts from upper-left corner
        '''
        if isinstance(arg, str):
            return self._get_rects_by_str(arg, reverse=reverse, origin_upper=origin_upper)
        return self._get_rects_by_inds(arg, origin_upper=origin_upper)

    ### auxiliary functions
    def _get_rects_by_str(self, s, reverse=False, origin_upper=False):
        '''
            real work to get rects for
                'row', 'col', 'all'
        '''
        valids=['row', 'col', 'all']
        assert s in valids, \
            'only allow str arg for `get_rects` ' \
            'in %s' % str(valids)

        ny, nx=self._ny, self._nx
        inds=np.arange(nx*ny).reshape(ny, nx)

        if s=='all':
            inds=np.ravel(inds)
        elif s=='col':
            inds=np.transpose(inds)

        if reverse:
            inds=np.flip(inds, axis=-1)

        return self._get_rects_by_inds(inds, origin_upper=origin_upper)

    def _get_rects_by_inds(self, inds, origin_upper=False):
        '''
            fetch rects in given indices
        '''
        return map_to_nested(self.__getitem__, inds,
                    is_scalar=self._is_type_item_index,
                    astype=None, kwargs=dict(origin_upper=origin_upper))

    ## support syntax: e.g. grid[0, 1], grid[:2, :]
    def _is_type_item_index(self, arg):
        '''
            whether `arg` is valid type for `__getitem__`
        '''
        return isinstance(arg, tuple) or \
               isinstance(arg, slice) or \
               isinstance(arg, numbers.Number)

    def __getitem__(self, prop, origin_upper=False):
        '''
            return rect via self[prop]
            always create new rect

            same syntax as GridSpec
            Note:
                g[0] for first rect
                    not g[0, :] in ndarray
        '''
        assert self._is_type_item_index(prop), \
            'unexpected type for index: %s' \
                % (type(prop).__name__)

        if isinstance(prop, numbers.Number):
            return self._get_ith_rect(prop, origin_upper=origin_upper)
        elif isinstance(prop, tuple):
            if len(prop)==1:
                prop=(prop[0], slice(None))
            elif len(prop)!=2:
                raise IndexError('unexpected len of indices for rect, '
                                 'only allow 1 or 2, '
                                 'but got %i' % len(prop))
        else:
            prop=(prop, slice(None))

        indy, indx=prop
        return self._get_rect(indx=indx, indy=indy, origin_upper=origin_upper)

    # Points
    def _standard_point_index(self, axis, i):
        '''
            standardize index of point
        '''
        num=self._get_num_rect_along_axis(axis)*2

        assert isinstance(i, numbers.Integral), \
                'only support integral for point index'

        i0=i
        if i<0:
            i+=num

        assert 0<=i<num, \
            'point %i is out of bounds for axis %s, ' \
            'at most %i points' % (i0, axis, num)

        return i

    def _get_ith_point(self, axis, i):
        '''
            return ith point along an axis

            In grid (nx, ny), along an axis, like 'x'
                there are 2*nx points, (x0, x1, ...)

            Parameters:
                axis: 'x' or 'y'
                    axis which the point is along

                i: int
                    order of the point to return
        '''
        return Point1D(self, axis, i)

    ## user method
    def get_point(self, axis, i):
        return self._get_ith_point(axis, i)
    get_point.__doc__=_get_ith_point.__doc__

    def get_all_points(self, axis):
        n=dict(x=self._nx, y=self._ny)[axis]
        return [self._get_ith_point(axis, i) for i in range(n)]

    def get_left(self, i):
        '''
            return left point of ith column
        '''
        return self._get_ith_point('x', 2*i)
    def get_right(self, i):
        '''
            return right point of ith column
        '''
        return self._get_ith_point('x', 2*i+1)
    def get_bottom(self, i):
        '''
            return bottom point of ith row
        '''
        return self._get_ith_point('y', 2*i)
    def get_top(self, i):
        '''
            return top point of ith row
        '''
        return self._get_ith_point('y', 2*i+1)

    # 1d distance
    def get_width(self, i=None):
        '''
            width of ith column

            if i is None, return total width
        '''
        if i is None:
            p0=self.get_left(0)
            p1=self.get_right(-1)
        else:
            p0=self.get_left(i)
            p1=self.get_right(i)
        return LineSeg1D(p0, p1)

    def get_height(self, i=None):
        '''
            height of ith row

            if i is None, return total height
        '''
        if i is None:
            p0=self.get_bottom(0)
            p1=self.get_top(-1)
        else:
            p0=self.get_bottom(i)
            p1=self.get_top(i)
        return LineSeg1D(p0, p1)

    def get_wspace(self, i):
        '''
            space between ith column and (i+1)th column
        '''
        p0=self.get_right(i)
        p1=self.get_left(i+1)
        return LineSeg1D(p0, p1)

    def get_hspace(self, i):
        '''
            space between ith row and (i+1)th row
        '''
        p0=self.get_top(i)
        p1=self.get_bottom(i+1)
        return LineSeg1D(p0, p1)

    def get_margin(self, axis, i):
        '''
            get left/right/bottom/top margin
            thats is space between grid and parent rect
        '''
        i=range(2)[i]

        p0=self._parent.get_point(axis, i)
        p1=self._get_ith_point(axis, [0, -1][i])

        if i==0:
            return LineSeg1D(p0, p1)
        else:
            return LineSeg1D(p1, p0)

    ## get collection of same type of dist
    def get_all_widths(self, origin_upper=False):
        '''
            return list of widths of columns
        '''
        return [self.get_width(i) for i in range(self._nx)]

    def get_all_heights(self, origin_upper=False):
        '''
            return list of heights of rows
        '''
        heights=[self.get_height(i) for i in range(self._ny)]
        if origin_upper:
            heights=heights[::-1]

        return heights

    def get_all_wspaces(self, origin_upper=False):
        '''
            return list of spaces between columns
        '''
        return [self.get_wspace(i) for i in range(self._nx-1)]

    def get_all_hspaces(self, origin_upper=False):
        '''
            return list of spaces between rows
        '''
        hspaces=[self.get_hspace(i) for i in range(self._ny-1)]
        if origin_upper:
            hspaces=hspaces[::-1]
        return hspaces

    def get_all_wmargins(self, origin_upper=False):
        '''
            return list of margins along x-axis
        '''
        if self._parent is None:
            return []
        return [self.get_margin('x', i) for i in range(2)]

    def get_all_hmargins(self, origin_upper=False):
        '''
            return list of margins along x-axis
        '''
        if self._parent is None:
            return []
        hmargins=[self.get_margin('y', i) for i in range(2)]
        if origin_upper:
            hmargins=hmargins[::-1]
        return hmargins

    # distance with val and unit
    def get_dist_unit(self, unit, axis='x', val=None):
        '''
            get `lncomb` for distance in an unit with optional `val`

            Parameters:
                val: None, float, dictable or list (as args)
                    val of distance in given `unit`

                    if None, return unit dist by default
                        for most unit, it means 1,
                            except for 'function' type unit
                                where to use empty dict {}

                    arg-able `val` only for 'func' type unit

                unit: str, int, or `LnComb`-like
                    unit of element in `vals`

                    lncomb-like: object with `to_lncomb`
                        `LnComb` for dist unit

                    str: 3 classes
                        - rect: 'figure', 'prect', 'grid', 'rect', 'recti'
                            w or h (by `axis`) of the rect

                            * 'figure': root rect in figure
                            * 'prect' : parent rect
                            * 'grid'  : this grid
                            * 'rect'  : origin rect (0th, bottom-left) in grid
                            * 'recti' : ith rect (origin at bottom-left) in grid

                        - absolute unit: 'pixel', 'points', ...
                            see `RectManager.get_size_unit`

                        - function: 'ticksep'
                            to call a function with given `axis` and `val`

                            only one currently for this type of unit

                            * 'ticksep[,k[=v]]': for `RectManager.get_ticksepsize_axis`
                                default kws (by order in funcion):
                                    dict(nchar_tick=1, nchar_lab=1,
                                         tick_out=False, lab=True)

                                optional k1[=v1],k2[=v2],...
                                    is to define order of args in function and default value
                                v could be string of int, float or 'true', 'false' (case-ignored)

                            if val is float, use it as first arg to call the func

                    int: ith rect in grid (origin at bottom-left)

                axis: 'x' or 'y'
                    along which axis for distance

                    it works only when 'rect' type unit given
        '''
        assert axis in ['x', 'y']

        # lncomb-like unit
        if hasattr(unit, 'to_lncomb'):
            if isinstance(unit, LineSeg1D):
                assert axis==unit._axis
            dist=unit.to_lncomb()
            if val is not None:
                if not isinstance(val, numbers.Real):
                    raise TypeError('only allow float for lncomb-like unit')
                dist=val*dist
            return dist

        # str for 'func', 'rect' type unit
        ## map for 'func' unit: {s: [func, list of keys]}
        s_funcs={'ticksep': [lambda g: g._get_manager().get_ticksepsize_axis,
                             ['nchar_tick', 'nchar_lab', 'tick_out', 'lab']],
                }
        if isinstance(unit, str):
            sunit, *kvitems=unit.split(',')
            # 'func' type unit
            if sunit in s_funcs:
                if val is None:
                    val={}

                lfunc, keys=s_funcs[sunit]
                func=lfunc(self)
                if isinstance(val, dict):
                    return func(axis, **val)
                else: # list or float `val`
                    if isinstance(val, numbers.Number):
                        vals=[val]
                    else:
                        vals=list(val)

                    nval=len(vals)
                    if nval>len(keys):
                        raise ValueError(f'too many args for unit `{sunit}`')

                    keys=list(keys)
                    kws_unit={}

                    # handle arg order and default val
                    n=len(kvitems)
                    for i in range(n):
                        k, *vs=kvitems[i].split('=', maxsplit=1)
                        k=k.strip()
                        if k not in keys:
                            s=f'unexpected arg in unit `{unit}`: `{k}`'
                            raise ValueError(s)

                        keys.remove(k)

                        # values
                        if i<nval:
                            kws_unit[k]=vals[i]
                        elif not vs:
                            continue
                        else:
                            s=vs[0].strip().lower()
                            if s in ['true', 'false']:
                                kws_unit[k]=True if s=='true' else False
                            elif re.match(r'^[+-]*\d+$', s):
                                kws_unit[k]=int(s)
                            else:
                                kws_unit[k]=float(s)

                    vals=vals[n:]
                    if vals:
                        for k, v in zip(keys, vals):
                            kws_unit[k]=v

                    return func(axis, **kws_unit)

            if kvitems:
                raise ValueError(f'unexpected unit: {unit}')

            # 'absolute' type
            if sunit in RectManager._UNITS:
                dist=self._get_manager().get_size_unit(sunit)
                if val is not None:
                    if not isinstance(val, numbers.Real):
                        raise TypeError('only allow float for \'absolute\' unit')
                    dist=dist*val
                return dist

            # 'rect' type
            if sunit=='figure':
                prect=self.get_manager().rect
            elif sunit=='prect':
                prect=self.get_parent()
            elif sunit=='grid':
                prect=self
            elif sunit[:4]=='rect':
                i=0 if not sunit[4:] else int(sunit[4:])
                prect=self.get_rect(i)
            else:
                raise ValueError(f'unexpected unit str: {unit}')

        # int unit for 'rect'
        else:
            if not isinstance(unit, numbers.Integral):
                raise TypeError('only allow str or int as unit')
            prect=self.get_rect(unit)

        ## val
        if val is None:
            val=1
        elif not isinstance(val, numbers.Real):
            raise TypeError('only allow float for non-func unit')

        assert axis in ['x', 'y']
        if axis=='x':
            return prect.width.to_lncomb()*val
        else:
            return prect.height.to_lncomb()*val

    ## get collection of dists in group
    _GROUPS_DIST=['width',   'height', 'wspace',  'hspace', 
                 'wmargin', 'hmargin', 'margin', 'sep']
    def get_dists_by_group(self, group='width', origin_upper=False):
        '''
            get group of dists

            valid group:
                'width', 'height',
                'wspace', 'hspace'
                'wmargin', 'hmargin'
                'sep': separations, wspace+hspace
                'margin': wmargin+hmargin
        '''
        assert group in self._GROUPS_DIST, \
                'only allow dist `group` in %s, ' \
                'but got [%s]' % (str(self._GROUPS_DIST), group)

        if group=='margin':
            d0=self.get_dists_by_group('wmargin', origin_upper=origin_upper)
            d1=self.get_dists_by_group('hmargin', origin_upper=origin_upper)
            return [*d0, *d1]

        if group=='sep':
            d0=self.get_dists_by_group('wspace', origin_upper=origin_upper)
            d1=self.get_dists_by_group('hspace', origin_upper=origin_upper)
            return [*d0, *d1]

        return getattr(self, 'get_all_%ss' % group)(origin_upper=origin_upper)

    # set linear constraints
    def set_dists_ratio(self, ratios, dist_group='width', origin_upper=False):
        '''
            set ratio for collection of dists

            Parameters:
                ratios: float or list of float
                    ratios between dists

                    len of ratios must be consistent with dist_group
                        that means n, or n-1 when `n` dists to set

                dist_group: 'width', 'height', 'wspace', 'hspace'
                           'wmargin', 'hmargin', 'margin', 'sep'
                    group of dists to set
        '''
        dists=self.get_dists_by_group(dist_group, origin_upper=origin_upper)
        if len(dists)<=1:
            return

        self._manager.set_dists_ratio(dists, ratios)

    def set_dists_val(self, vals, dist_group='width', units=None, origin_upper=False):
        '''
            set value(s) for collection of dists

            Parameters:
                vals: float, dict, LnComb-like object, or list of None, float, arg-like scalar
                    value(s) to set

                    if float, dict or LnComb-like obj, expand to `[v]*n` for n = num of dists

                    if list, must has same len with num of dists
                        elements could be of None, float, dict or list of args
                            the last two valid only when `units` not None

                        if None, skip corresponding dist

                dist_group: 'width', 'height', 'wspace', 'hspace'
                           'wmargin', 'hmargin', 'margin', 'sep'
                    group of dists to set

                units: None, str, int, `LnComb`-like, or list of None, scalar
                    unit scalar:
                        str: 'figure', 'prect', 'grid', 'rect', or
                             'points', 'pixel', ... or
                             'ticksep'
                        int: ith rect in grid (count from bottom-left)
                    see `get_dist_unit` for detail

                    if list, its length must be equal to num of dists
                        elements could be None or scalar
                        if None, use no unit
        '''
        dists=self.get_dists_by_group(dist_group, origin_upper=origin_upper)
        n=len(dists)

        # values
        if isinstance(vals, numbers.Number) or \
           isinstance(vals, dict) or \
           hasattr(vals, 'to_lncomb'):
            vals=[vals]*n
        else:
            vals=list(vals)
            assert len(vals)==n, \
                f'cannot assign {len(vals)} vals to {len(dists)} dists'

        # units
        if units is not None:
            if isinstance(units, str) or \
               isinstance(units, numbers.Number) or \
               hasattr(units, 'to_lncomb'):
                units=[units]*n
            assert len(units)==n, \
                f'cannot assign {len(units)} units to {len(dists)} dists'

            vals1=[]
            for di, vi, ui in zip(dists, vals, units):
                if vi is None:
                    vals1.append(vi)
                    continue

                if ui is None:
                    if not isinstance(vi, numbers.Real):
                        raise TypeError('only allow float for non-unit value')
                    vals1.append(vi)
                    continue

                vi=self.get_dist_unit(ui, axis=di._axis, val=vi)
                vals1.append(vi)
            vals=vals1

        # set vals
        for di, vi in zip(dists, vals):
            if vi is None:
                continue
            di.set_to(vi)

    def set_dists_bound(self, val, dist_group='sep', upper=True, origin_upper=False):
        '''
            set dists' bound
        '''
        dists=self.get_dists_by_group(dist_group, origin_upper=origin_upper)
        self._manager.set_dists_bound(val, *dists, upper=upper)

    def set_dists_le(self, maxd, dist_group='width', origin_upper=False):
        '''
            set group of dists less-or-equal to val

            :param dist_group: str
                dist group
                    'width', 'height', 'wspace', 'hspace'
                    'wmargin', 'hmargin', 'margin', 'sep'
                see `get_dists_by_group` for detail
        '''
        self.set_dists_bound(maxd, dist_group, upper=True, origin_upper=origin_upper)

    def set_dists_ge(self, mind, dist_group='sep', origin_upper=False):
        '''
            set group of dists greater-or-equal to val
            usually used to set not too small separation

            :param dist_group: str
                dist group
                    'width', 'height', 'wspace', 'hspace'
                    'wmargin', 'hmargin', 'margin', 'sep'
                see `get_dists_by_group` for detail
        '''
        self.set_dists_bound(mind, dist_group, upper=False, origin_upper=origin_upper)

    def set_dists_lim(self, lim, dist_group='sep', origin_upper=False):
        '''
            set lim for group of dists 

            :param dist_group: str
                dist group
                    'width', 'height', 'wspace', 'hspace'
                    'wmargin', 'hmargin', 'margin', 'sep'
                see `get_dists_by_group` for detail
        '''
        dists=self.get_dists_by_group(dist_group, origin_upper=origin_upper)
        self._manager.set_dists_lim(lim, *dists, upper=upper)

    def set_dists_equal(self, dist_group='bbox'):
        '''
            set collection of dists equal

            Parameters:
                dist_group: 'width', 'height', 'wspace', 'hspace'
                           'wmargin', 'hmargin', 'margin', 'sep'
                           or 'x', 'y', 'xy', 'bbox'

                    group of dist to set equal

                    if 'x', 'y', 'xy',
                        set
                            all dists along one or both axis
                                with same type
                        equal

                    if 'bbox',
                        set
                            all dists inside bbox
                                that are width, height, wspace, hspace
                        equal

        '''
        if dist_group=='xy':
            self.set_dists_equal('x')
            self.set_dists_equal('y')
            return

        if dist_group in ['x', 'y']:
            s=dict(x='width', y='height')[dist_group]

            self.set_dists_equal(s)
            self.set_dists_equal('%sspace' % s[0])
            self.set_dists_equal('%smargin' % s[0])

            return

        if dist_group=='bbox':
            self.set_dists_equal('width')
            self.set_dists_equal('wspace')

            self.set_dists_equal('height')
            self.set_dists_equal('hspace')

            return

        self.set_dists_ratio(1, dist_group)

    def set_all_dists_equal(self):
        '''
            set all dists in same group equal
                including width, height, wspace, hspace, wmargin, hmargin
        '''
        self.set_dists_equal('xy')

    ## global layout
    def set_rect_sep_margin_ratio(self, ratios, axis='both', origin_upper=False):
        '''
            ratio between size of
                    base rect,
                    first separation (if exists) between rect, and
                    first margin

            3 ratios must be given, even no sep
                like nx=1, no wspace

            Parametes:
                ratios: float, or list of float
                    ratios between dists
                        corresponding to [rect, sep, margin]

                    if list, must has len 2 or 3
                        if len=2, first 1 is omit

                axis: 'x', 'y', 'both', 'xy'
                    axis of the dists

                    if 'both', set same ratios to both axis
        '''
        if axis in ['both', 'xy']:
            self.set_rect_sep_margin_ratio(ratios, 'x', origin_upper=origin_upper)
            self.set_rect_sep_margin_ratio(ratios, 'y', origin_upper=origin_upper)
            return

        assert axis in list('xy'), \
                "only support 'x', 'y' for axis"

        # dists
        s=dict(x='width', y='height')[axis]

        if axis=='y':
            ind=-1 if origin_upper else 0

        dists=[getattr(self, 'get_'+s)(ind)]

        n=getattr(self, '_n'+axis)
        if n>1:
            dists.append(getattr(self, f'get_{s[0]}space')(ind))

        dists.append(self.get_margin(axis, ind))

        # ratios
        ratios=list(_parse_ratios(ratios, 3))
        if len(dists)<3:
            ratios.pop(-2)

        self._manager.set_dists_ratio(dists, ratios)

    def set_no_overlap(self, min_dist=0, inbox=True):
        '''
            set no overlap in grid, that means
                width, height of all rects >= 0
                wspaces, hspaces >= 0

            if `inbox` is True, means to set grid in parent box
                then set all margins >= 0
        '''
        self.set_dists_bound(min_dist, 'width', upper=False)
        self.set_dists_bound(min_dist, 'height', upper=False)
        self.set_dists_bound(min_dist, 'sep', upper=False)

        if inbox:
            self.set_dists_bound(min_dist, 'margin', upper=False)

    def set_inbox(self, axis='both'):
        '''
            set all points in parent box
        '''
        if self._parent is None:
            return

        vs_axis=['x', 'y', 'both']
        assert axis in vs_axis, 'ony allow `axis` in %s' % str(vs_axis)

        if axis=='both':
            self.set_inbox(axis='x')
            self.set_inbox(axis='y')

            return

        points=self.get_all_points(axis)

        box0=self._parent
        p0, p1=box0.get_point(axis, 0), box0.get_point(axis, -1)

        for p in points:
            self._manager.set_dists_nonneg(p-p0, p1-p)

    ## width/height of grid
    def set_width(self, w):
        '''
            set total width
        '''
        self.width.set_to(w)

    def set_height(self, h):
        '''
            set total height
        '''
        self.height.set_to(h)

    ## widths/heights of rects
    def set_widths_equal(self):
        '''
            set widths of all rects equal
        '''
        self.set_dists_equal(dist_group='width')

    def set_heights_equal(self):
        '''
            set widths of all rects equal
        '''
        self.set_dists_equal(dist_group='height')

    def set_rects_equal(self, axis='xy'):
        '''
            set length of rects equal along an/both axis
        '''
        if axis=='xy':
            self.set_widths_equal()
            self.set_heights_equal()
            return

        assert axis in ['x', 'y']
        if axis=='x':
            self.set_widths_equal()
        else:
            self.set_heights_equal()

    ## constraint to separations
    def set_seps(self, vals, axis='both', units=None, origin_upper=False):
        '''
            set separations for vals

            Parameters:
                vals: float, dict, or list of None, float, dict or list of args
                    value(s) to set for separations between rects

                    if float or dict, expand to `[v]*n` for n = `nx-1` or `ny-1`

                    if list, its len must equal to `nx-1` or `ny-1`
                        elements could be of None, float, dict or list of args
                            the last two valid only when `units` not None

                        if None, skip corresponding dist

                    NOTE: if `axis=='both'`, same `vals` for both x/y-axis
                        use `RectGrid.set_dists_val` with `dist_group='sep'`
                            to set different separations along x and y axis

                axis: 'x', 'y', or 'both'
                    separations along which axis to set

                    if 'both', same `vals` for both x/y-axis

                units: None, str, int, `LnComb`-like, or list of None, scalar
                    unit scalar:
                        str: 'figure', 'prect', 'grid', 'rect', or
                             'points', 'pixel', ... or
                             'ticksep'
                        int: ith rect in grid (count from bottom-left)
                        lncomb-like obj: with attr `to_lncomb`
                    see `get_dist_unit` for detail

                    if list, its length must be equal to `nx-1` or `ny-1`
                        elements could be None or scalar
                        if None, use no unit, that means 'inches' by default
        '''
        if axis=='both':
            self.set_seps(vals, 'x', units)
            self.set_seps(vals, 'y', units)
            return

        dist_group=dict(x='wspace', y='hspace')[axis]

        self.set_dists_val(vals, dist_group=dist_group, units=units, origin_upper=origin_upper)

    def set_seps_zero(self, axis='both'):
        '''
            set separations between rect zero

            Parameters:
                axis: 'x', 'y', or 'both'
                    separations along which axis to set
        '''
        self.set_seps(0, axis=axis)

    def set_seps_ratio_to(self, dist0, ratios, axis='both', origin_upper=False):
        '''
            set ratios of separations with respect to a base distance

            Parameters:
                dist0: LineSeg1D, RectGrid, Rect
                    base distance

                ratios: float, or array of float
                    ratio of sep to `dist0`

                axis: 'x', 'y', or 'both'
        '''
        if axis=='both':
            self.set_seps_ratio_to(dist0, ratios, 'x', origin_upper=origin_upper)
            self.set_seps_ratio_to(dist0, ratios, 'y', origin_upper=origin_upper)
            return

        assert axis in list('xy'), \
                "only support 'x', 'y', 'both' for axis"

        # dist0
        if not isinstance(dist0, LineSeg1D):
            assert isinstance(dist0, RectGrid) or isinstance(dist0, Rect)
            p0, p1=[dist0.get_point(axis, i) for i in [0, -1]]
            dist0=p1-p0

        s=dict(x='wspace', y='hspace')[axis]
        dists=getattr(self, f'get_all_{s}s')(origin_upper=origin_upper)

        if isinstance(ratios, numbers.Number):
            ratios=[ratios]*len(dists)
        else:
            assert len(ratios)==len(dists)

        self._manager.set_dists_ratio([dist0, *dists], [1, *ratios])

    ## margins
    def set_margins(self, vals, axis='both', units=None, origin_upper=False):
        '''
            set margins for vals

            Parameters:
                vals: float, dict, or list of None, float, dict or list of args
                    value(s) to set for margins between rect and its parent

                    if float or dict, expand to `[v]*n` for n = num of dists

                    if list, its len must equal to `nx-1` or `ny-1`
                        elements could be of None, float, dict or list of args
                            the last two valid only when `units` not None

                        if None, skip corresponding dist

                    NOTE: if `axis=='both'`, same `vals` for both x/y-axis
                        use `RectGrid.set_dists_val` with `dist_group='sep'`
                            to set margins along x and y axis

                axis: 'x', 'y', or 'both'
                    margins along which axis to set

                    if 'both', same `vals` for both x/y-axis

                units: None, str, int, obj with `to_lncomb`, or list of scalar
                    unit scalar:
                        str: 'figure', 'prect', 'grid', 'rect', or
                             'points', 'pixel', ... or
                             'ticksep'
                        int: ith rect in grid (count from bottom-left)
                        lncomb-like obj: with attr `to_lncomb`

                    see `get_dist_unit` for detail
        '''
        if axis=='both':
            self.set_margins(vals, 'x', units)
            self.set_margins(vals, 'y', units)
            return

        dist_group=dict(x='wmargin', y='hmargin')[axis]
        self.set_dists_val(vals, dist_group=dist_group, units=units, origin_upper=origin_upper)

    def set_margins_zero(self, axis='both'):
        '''
            set margins to parent rect zero

            Parameters:
                axis: 'x', 'y', or 'both'
                    margins along which axis to set
        '''
        self.set_margins(0, axis=axis)

    def set_grid_center(self):
        '''
            set grid locating in center of parent rect
        '''
        self.set_dists_equal(dist_group='margin')

    ### location at a top rect
    def set_loc_at_toprect_along_axis(self, toprect, axis,
                                            loc, locing='m', locunits=None):
        '''
            set loc at `toprect` along an axis
            base function for 'location setting'

            Parameters:
                toprect: `Rect` instance
                    top rect what the `loc` is relative to

                axis: 'x', 'y'
                    along which axis the loc is

                loc: float, dict, (x0, x1)
                    location in top rect

                    if float or dict, means margin
                        `locing` not work in this case (fixed to 'm')

                    meaning of x0, x1 defined by `locing` and `locunits`

                    x0, x1 could be None, float, or dict or list of args
                        the last two valid only when `locunits` is of 'func' type
                            see `get_dist_unit` for detail
                        if None, skip setting

                locing: 'x', 'w', 'm'
                    how the `loc` specifies location
                    fixed to 'm' when float given in `loc` or x1 unit being 'ticksep'

                    'x': points x
                        x0, x1 - two endpoints of grid along the axis

                    'w': width
                        x0 - left points
                        x1 - width

                    'm': margin
                        x0, x1 - two margins to top rect

                locunits: None, str, int, `LnComb`-like, or list of scalar
                    what the value of number in `loc` really is

                    if None, use width or height (by `axis`) as default

                    if unit scalar,
                        str: 'figure', 'prect', 'grid', 'rect', or
                             'points', 'pixel', ... or
                             'ticksep[,k=[v]]'
                        int: ith rect in grid (count from bottom-left)
                        lncomb-like obj: with attr `to_lncomb`

                    see `get_dist_unit` for detail

                    if scalar, use same unit for both x0 and x1
        '''
        # type check
        if not isinstance(toprect, Rect):
            raise TypeError('only allow <Rect> instance as `toprect`')

        if toprect._get_manager() is not self._get_manager():
            raise ValueError('must same `RectManager` for `toprect`')

        if axis not in ['x', 'y']:
            raise ValueError(f"only allow 'x' or 'y' as `axis`, but got {axis}")

        assert loc is not None
        scalar_loc=isinstance(loc, numbers.Number) or isinstance(loc, dict)

        ## locing
        if locing not in ['x', 'w', 'm']:
            raise ValueError(f'unexpected `locing`: {locing}')

        ## locunits
        if locunits is None:
            locunits=toprect.width if axis=='x' else toprect.height
            locunits=(locunits, locunits)
        elif isinstance(locunits, str) or \
             isinstance(locunits, numbers.Number) or \
             hasattr(locunits, 'to_lncomb'):
            if scalar_loc:  # all scalar
                locunits=self.get_dist_unit(locunits, axis=axis, val=loc)
                loc=1
            elif isinstance(locunits, str) and locunits[:7]=='ticksep':
                locing='m'
            locunits=(locunits, locunits)
        elif len(locunits)!=2:
            raise ValueError(f'len of `locunits` must be 2 if list, '
                             f'but got {len(locunits)}')
        elif isinstance(locunits[1], str) and locunits[1][:7]=='ticksep':
            locing='m'

        ## loc
        if scalar_loc:
            loc=(loc, loc)
            locing='m'
        elif len(loc)!=2:
            raise ValueError(f'non scalar `loc` must has len 2, '
                             f'but got {len(loc)}')

        # line segs to constraint by `locing`
        pp0=toprect.get_point(axis, 0)
        p0=self.get_point(axis, 0)
        p1=self.get_point(axis, -1)

        dist0=p0-pp0
        if locing=='x':
            dist1=p1-pp0
        elif locing=='w':
            dist1=p1-p0
        else:  # locing=='m'
            pp1=toprect.get_point(axis, -1)
            dist1=pp1-p1

        dists=[dist0, dist1]  # distances to equal to x0, x1

        # set location
        manager=self.get_manager()
        for di, vi, ui in zip(dists, loc, locunits):
            if vi is None:
                continue
            vi=self.get_dist_unit(ui, axis=axis, val=vi)
            manager._add_lncomb(di, vi)

    def set_loc_along_axis(self, axis, loc, at='parent', **kwargs):
        '''
            set location at a rect along an axis

            :param axis: 'x' or 'y'

            :param at: str or `Rect` instance, default: 'parent'
                specify which rect is for the `loc`

                str: 'parent', 'root', 'figure', 'fig'
                    last 3 all for root rect of figure

            other arguments refer to `set_loc_at_toprect_along_axis` for detail
        '''
        prect=self._get_rect_at_for_loc(at)
        self.set_loc_at_toprect_along_axis(prect, axis, loc, **kwargs)

    def set_loc(self, loc, locing='wh', locunits=None, at='parent'):
        '''
            set loc at parent/root rect

            Parameters:
                loc: float, dict, 4 or 2-tuple of floats
                    fraction at a rect

                    if float or dict, means margin
                        `locing` fixed to 'margin'

                    if len==2, use same loc for x-/y-axis

                    if len==4, [x0, y0, x1, y1]
                        meaning of xi, yi is given by `locing` and `locunits`

                    elements could be None to ignore it

                locing: 'wh', 'xy', 'margin', 'm', or combination of 'xwm', default: 'wh'
                    how the `loc` specifies location
                    fixed to 'm' when float given in `loc`

                    'm' is same as 'margin'

                    combination of 'xwm': used to specify different locing along x/y axis
                        'x', 'w', 'm' for points, width, and margin
                        see `set_loc_at_toprect_along_axis` for detail

                locunits: None, str, int, `LnComb`-like, or list
                    what the value of number in `loc` really is

                    if None, use width or height (by `axis`) as default

                    if list, len must be 1, 2, 4
                        - len==1: element for both x and y axis
                        - len==2: elements for x, y respectively
                        - len==4: correspond to [x0, y0, x1, y1]

                    if unit scalar,
                        str: 'figure', 'prect', 'grid', 'rect', or
                             'points', 'pixel', ... or
                             'ticksep[,ki=[vi]]'
                        int: ith rect in grid (count from bottom-left)
                        lncomb-like obj: with attr `to_lncomb`

                    see `get_dist_unit` for detail

                    if scalar, use same unit for both x0 and x1

                at: str or `Rect` instance, default: 'parent'
                    specify which rect is for the `loc`

                    str: 'parent', 'root', 'figure', 'fig'
                        last 3 all for root rect of figure
        '''
        # args normalize
        prect=self._get_rect_at_for_loc(at)

        ## loc: to xloc, yloc
        if isinstance(loc, numbers.Number) or isinstance(loc, dict):
            xloc=yloc=loc
            locing='m'
        else:
            loc=list(loc)
            n=len(loc)
            if n==2:
                xloc=yloc=loc
            elif n==4:
                x0, y0, x1, y1=loc
                xloc=[x0, x1]
                yloc=[y0, y1]
            else:
                raise ValueError(f'len of `loc` must be 2 or 4 if list, '
                                 f'but got {len(loc)}')

        ## locing: to xlocing, ylocing
        if locing in ['wh', 'xy', 'margin', 'm']:
            xlocing=ylocing=locing[0]
        elif len(locing)==2:
            xlocing, ylocing=locing
        else:
            raise ValueError(f'unexpected value for `locing`: {locing}')

        ## locunits: to xlocu, ylocu
        if locunits is None:
            xlocu=prect.width
            ylocu=prect.height
        elif isinstance(locunits, str) or \
             isinstance(locunits, numbers.Number) or \
             hasattr(locunits, 'to_lncomb'):
            xlocu=ylocu=locunits
        else:
            locunits=list(locunits)
            n=len(locunits)
            if n==1:
                xlocu=ylocu=locunits[0]
            elif n==2:
                xlocu, ylocu=locunits
            elif n==4:
                ux0, uy0, ux1, uy1=locunits
                xlocu=[ux0, ux1]
                ylocu=[uy0, uy1]
            else:
                raise ValueError(f'len of `locunits` must be 1, 2 or 4 if list, '
                                 f'but got {len(locunits)}')

        # set loc for both axis individually
        self.set_loc_at_toprect_along_axis(prect, 'x', xloc,
                                                locing=xlocing, locunits=xlocu)
        self.set_loc_at_toprect_along_axis(prect, 'y', yloc,
                                                locing=ylocing, locunits=ylocu)

    ### auxiliary functions
    def _get_rect_at_for_loc(self, at):
        '''
            return rect for loc to relate

            :param at: str or `Rect` instance, default: 'parent'
                specify which rect is for the `loc`

                str: 'parent', 'root', 'figure', 'fig'
                    last 3 all for root rect of figure
        '''
        if isinstance(at, Rect):
            prect=at
        else:
            if not isinstance(at, str):
                raise TypeError(f'unexpected type for `at`: {type(at)}')

            if at not in ['parent', 'root', 'figure', 'fig']:
                raise ValueError(f'unexpected value for `at`: {at}')

            if at=='parent':
                prect=self.get_parent()
            else:
                prect=self.get_manager().rect

        return prect

    ## set all dists in grid
    def set_grid_dists(self, loc=0.1, locing='wh', locunits=None,
                            origin_upper=False,
                            ratios_w=1, ratios_h=1,
                            wspaces=0, wpunits=None,
                            hspaces=0, hpunits=None,
                            ratio_wh=None):
        '''
            Set distances in grid relative that of origin rect

            =========
            To locate axes in grid, it is enough to specify
                loc: relative to whole axes in parent rect
                ratios: of width/height among axes
                ratios: between w and h of one ax, aka aspect
                wspaces/hspaces: between nearby axes
            among which
                - ratios of w/h are dimensionless value
    
                - loc and w/hspaces are some kind of distances
                    besides the value, they must have some unit
                        like w/h of figure or base axes by default
    
            If all setted, rects of grid should be fixed relative to parent rect
    
            ========
            Parameters:
                loc, locing, locunits, at: args to set location at parent
                    see `RectGrid.set_loc` for detail

                origin_upper: bool, default False
                    whether the index is given with origin in upper

                    if True, order of axes starts from upper-left corner
                    otherwise, bottom-left

                wspaces, wpunits: kwargs to specify wspaces
                hspaces, hpunits: kwargs to specify hspaces
                    value and unit for distance of wspaces/hspaces

                    see `RectGrid.set_seps` for detail

                ratios_w, ratios_h: None, float, array of float
                    ratios of width/height of rects in grid relative to origin rect

                    if None, not set

                    if float, it means ratios of other rects to origin rect

                    if array, its len should be `nx-1`, or `nx` (`ny-1`, `ny` respectively)
                        for `nx-1`, it means `[1, *ratios]`

                ratio_wh: None, float, or tuple (int, float), ((int, int), float), (None, int)
                    ratio w/h for one axes or whole axes region (if given (None, int))

                    if None, not set
                    if float, set for 0th rect
                    if (i, r) or ((i, j), r), set for rect in index i or (i, j)
                    if (None, r), set for whole rect
        '''
        # location of grid
        self.set_loc(loc, at='parent', locing=locing, locunits=locunits)

        ## wspaces/hspaces
        wpunits=0 if wpunits is None else wpunits  # use first rect by default
        hpunits=0 if hpunits is None else hpunits

        self.set_seps(wspaces, axis='x', units=wpunits, origin_upper=origin_upper)
        self.set_seps(hspaces, axis='y', units=hpunits, origin_upper=origin_upper)

        # ratio of widths/heights with respect to axes[0, 0] at left-bottom
        if ratios_w is not None:
            self.set_dists_ratio(ratios_w, 'width', origin_upper=origin_upper)

        if ratios_h is not None:
            self.set_dists_ratio(ratios_h, 'height', origin_upper=origin_upper)

        ## ratio of w/h
        if ratio_wh is not None:
            if isinstance(ratio_wh, numbers.Number):
                ind=(-1, 0) if origin_upper else (0, 0)
            else:
                ind, ratio_wh=ratio_wh

            if ind is None:
                region=self
            else:
                region=self.get_rect(ind)

            manager=self._get_manager()
            manager.set_ratio_to([region.width, region.height], [ratio_wh, 1])

    ## properties
    for k_ in _GROUPS_DIST:
        doc_='''
                {0}s of grid

                return a special list to
                    support syntax for dist set, e.g.:
                        grid.{0}s[0]=1
                        grid.{0}s[:]=1
                        grid.{0}s[:2]=1, 2
            '''.format(k_)

        locals()[k_+'s']=property(
            eval('lambda self: SetterList(self.get_dists_by_group(%s))'
                    % repr(k_)),
            doc=doc_)
    del k_, doc_

    width =property(lambda self: self.get_width(), doc='total width')
    height=property(lambda self: self.get_height(), doc='total height')

    ## getattr, setattr: for some abbreviations
    def _is_abbr_attr(self, attr):
        '''
            whether a valid abbreviated attribute
                x, y: point along x/y-axis
                wh: width and height
                a: axes
                    used to get ith rect,
                        starting point to create axes

            if not support, return False
        '''
        if attr[0] in list('xywha') and attr[1:].isdigit():
            return True
        return False

    def __getattr__(self, prop):
        if self._is_abbr_attr(prop):
            a, i=prop[0], int(prop[1:])
            if a=='w':
                return self.get_width(i)
            elif a=='h':
                return self.get_height(i)
            elif a=='a':  # a for axes, used get ith rect
                return self.get_rect(i)

            return self.get_point(a, i)

        raise AttributeError('unexpected attr: %s' % prop)

    def __setattr__(self, prop, val):
        if self._is_abbr_attr(prop) and prop[0]!='a':
            t=getattr(self, prop)
            t.set_to(val)
            return
        super().__setattr__(prop, val)

    # create axes
    def create_axes(self, rects='row', origin_upper=False,
                            sharex=False, sharey=False,
                            return_mat=True, squeeze=True, **kwargs):
        '''
            create axes in grid

            Parameters:
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
        '''

        isarr_rects=rects in ['row', 'col']  # rects could be arranged in array
        rects=self.get_rects(rects, origin_upper=origin_upper)

        # sharex, sharey
        kws1={}
        if sharex:
            if type(sharex) is bool:
                sharex='all'
            kws1['sharex']=self.get_rects(sharex)
        if sharey:
            if type(sharey) is bool:
                sharey='all'
            kws1['sharey']=self.get_rects(sharey)

        # create axes
        manager=self._get_manager()
        fig, axes=manager.create_axes_in_rects(rects, **kws1, **kwargs)

        ## to matrix
        if isarr_rects and return_mat:
            axes=np.asarray(axes)
            if squeeze and (self._nx==1 or self._ny==1):
                axes=np.ravel(axes)
                if len(axes)==1:
                    axes=axes[0]

        return fig, axes

    # to string
    def __repr__(self):
        '''
            representation in string

            return RectGrid(nrows, ncols)
                similar as `matplotlib.GridSpec`
        '''
        return '%s(%i, %i)' % (type(self).__name__,
                               self._ny, self._nx)

class Rect:
    '''
        class for rectangle
            starting point to create axes
    '''
    def __init__(self, grid, indy=0, indx=0, yspan=1, xspan=1):
        '''
            init of rectangle

            Parameters:
                grid: RectGrid instance
                    grid which rectangle locates at

                indx, indy: int
                    index of rectangle in grid

                xspan, yspan: int
                    span of rect in grid
        '''
        # rect grid
        assert isinstance(grid, RectGrid), \
                'only type RectGrid for rect grid, '\
                'but got %s' % (type(grid).__name__)
        self._grid=grid

        # index in grid, standardize
        x0, y0=grid._standard_rect_xyind(indx=indx, indy=indy)

        # span
        assert xspan>=1 and yspan>=1, \
                'only allow positive for span'
        x1, y1=grid._standard_rect_xyind(indx=x0+xspan-1,
                                         indy=y0+yspan-1)

        self._indx0, self._indy0=x0, y0
        self._indx1, self._indy1=x1, y1

        # attributes for axes
        self._axes=None

    # index in grid
    def is_left(self):
        '''
            whether left in grid
        '''
        return self._indx0==0

    def is_bottom(self):
        '''
            whether bottom in grid
        '''
        return self._indy0==0

    def get_xyind_sw(self):
        '''
            xy-index of left-bottom corner
            SW: south-west
        '''
        return self._indx0, self._indy0

    def get_xyspan(self):
        '''
            xy-span
        '''
        return self._indx1-self._indx0+1, \
               self._indy1-self._indy0+1

    @property
    def area(self):
        '''
            return (ix1-ix0+1)*(iy1-iy0)+1
        '''
        w, h=self.get_xyspan()
        return w*h

    @property
    def order(self):
        '''
            order in grid
            only support when area=1
        '''
        assert self.area==1, \
            'only support property `order` for area==1'
        return self._grid._standard_rect_xyind_arg(
            indx=self._indx0, indy=self._indy0, return_order=True)

    # manager
    def _get_manager(self):
        '''
            return manger of grid

            no register work done
                do it explicitly via RectGrid
            return None if not exists
        '''
        return self._grid._manager

    get_manager=_get_manager

    # grid
    def _add_grid(self, nx, ny):
        '''
            add grid inside the rect
        '''
        return RectGrid(nx=nx, ny=ny,
                        parent=self,
                        manager=self._grid._get_manager())

    ## user methods
    def add_grid(self, nx=1, ny=1):
        '''
            add grid inside rect

            Parameters:
                nx, ny: int
                    ncols and nrows of grid
        '''
        return self._add_grid(nx=nx, ny=ny)

    # Points
    def get_point(self, axis, i):
        '''
            return point in rect

            a rect is specified by 4 variables:
                x0, x1, y0, y1

            Parameters:
                axis: 'x' or 'y'
                    axis of the point

                i: 0, 1 or -1
                    index of point along an axis
        '''
        assert axis in list('xy')

        i=[0, 1][i]

        # index in grid
        indg=2*getattr(self, '_ind%s%i' % (axis, i))+i

        return self._grid._get_ith_point(axis, indg)

    def get_left(self):
        # left point
        return self.get_point('x', 0)
    def get_right(self):
        # right point
        return self.get_point('x', 1)
    def get_bottom(self):
        # bottom point
        return self.get_point('y', 0)
    def get_top(self):
        # right point
        return self.get_point('y', 1)

    # Distances
    def get_width(self):
        # width
        return LineSeg1D(self.get_left(), self.get_right())
    def get_height(self):
        # height
        return LineSeg1D(self.get_bottom(), self.get_top())

    # linear constraint
    def set_left(self, p):
        '''
            set left aligned with another point

            Parameters:
                p: float, Point1D, or LnComb-like
                    point to align with

                    if float, means point in root
                        with a distance to root edge,
                            in unit of 'inches'
        '''
        self._get_manager()._add_lncomb(self.get_left(), p)
    def set_right(self, p):
        '''
            set right aligned with another point
        '''
        self._get_manager()._add_lncomb(self.get_right(), p)
    def set_bottom(self, p):
        '''
            set bottom aligned with another point
        '''
        self._get_manager()._add_lncomb(self.get_bottom(), p)
    def set_top(self, p):
        '''
            set top aligned with another point
        '''
        self._get_manager()._add_lncomb(self.get_top(), p)

    def set_width(self, d):
        '''
            set width equal to another distance

            Parameters:
                d: float, LineSeg1D, or LnComb-like
                    distance to set current width

                    if float, in unit of 'inches'
        '''
        self._get_manager()._add_lncomb(self.get_width(), d)
    def set_height(self, d):
        '''
            set height equal to another distance
        '''
        self._get_manager()._add_lncomb(self.get_height(), d)

    ## ratios
    def set_height_to_width(self, hw):
        '''
            set height ratio to width

            Parameters:
                hw: float
                    ratio h/w
        '''
        dists=[self.get_width(), self.get_height()]
        ratios=[1, hw]
        self._get_manager().set_dists_ratio(dists, ratios)

    def set_aspect(self, hw):
        '''
            set aspect ratio, h/w

            alias of `set_height_to_width`
        '''
        self.set_height_to_width(hw)

    def set_width_to_height(self, wh):
        '''
            set width ratio to height
        '''
        self.set_height_to_width(1/wh)

    def set_aspect_equal(self):
        '''
            set equal aspect ratio
        '''
        self.set_aspect(1)

    ### some aliases
    set_hw_ratio=set_height_to_width
    set_wh_ratio=set_width_to_height

    ## properties
    for k_ in 'left right bottom top width height'.split():
        g_=locals()['get_'+k_]
        s_=locals()['set_'+k_]
        locals()[k_]=property(g_, s_)
    del k_, g_, s_

    # x0, x1=left, right
    # y0, y1=bottom, top

    ## getattr/setattr: rect.x0, rect.x1, rect.y0, rect.y1
    def __getattr__(self, prop):
        if prop[0] in ['x', 'y'] and prop[1:].isdigit():
            a, i=prop[0], int(prop[1:])
            return self.get_point(a, i)

        raise AttributeError('unexpected attr: %s' % prop)

    def __setattr__(self, prop, val):
        if prop[0] in ['x', 'y'] and prop[1:].isdigit():
            p=self.__getattr__(prop)
            return self._get_manager()._add_lncomb(p, val)

        return super().__setattr__(prop, val)

    # other frequently-used points
    def get_partition(self, axis, p):
        '''
            partition point along an axis
            return LnComb instance

            (1-p)*p0+p*p1
            p0, p1:
                left, right for x-axis
                bottom, top for y-axis
        '''
        p0=self.get_point(axis, 0)
        p1=self.get_point(axis, 1)
        return (1-p)*p0+p*p1

    def get_center(self, axis):
        '''
            center point along an axis
            return LnComb instance
        '''
        return self.get_partition(axis, 0.5)

    # locate of rect
    def align_with(self, a, axis='x', on=0, on2=None):
        '''
            align with another rect or point

            Parameters:
                a: Rect, Point1D, or LnComb-like
                    another rect to align with

                axis: 'x', 'y'
                    along which axis to align

                on: int, float, or str
                    anchor for alignment

                    if int, only 0, 1, or 11
                        corresponding to 2 edges in an axis
                            11 for both edges

                    if float, means partition point

                    if str, only 'both', or
                         'left', 'right' for x-axis,
                         'bottom', 'top' for y-axis

                on2: None, or same type with `on` (except 'both' or 11)
                    anchor in second rect

                    if None, use `on`

                    if `on` is 11 or 'both' to align 2 edges
                        `on2` must be None
        '''
        # type check and standardize
        on=self._check_anchor_type(on, axis)

        if on=='both':
            assert on2 is None, \
                'set `on2` to None when set both edges'

            self.align_with(a, axis, 0)
            self.align_with(a, axis, 1)

            return

        p0=self.get_partition(axis, on)

        # another anchor
        if isinstance(a, Rect):
            if on2 is None:
                on2=on
            else:
                on2=self._check_anchor_type(on2, axis)

                assert on2!='both', \
                    'not allow \'both\' for `on2`'
            p1=a.get_partition(axis, on2)
        else: # align with point
            assert on2 is None, \
                'no anchor for aligning with point'
            p1=a

        self._get_manager().align_points(p0, p1)

    ## auxiliary functions
    def _check_anchor_type(self, anchor, axis):
        '''
            check type of argument `on`
                in method `align_with`

            also standardize to 0, 1, float or 'both'
        '''
        assert axis in list('xy'), \
            'only support x or y axis, ' \
            'but got '+str(axis)

        if isinstance(anchor, numbers.Number):
            if anchor==11:
                return 'both'

            assert anchor in [0, 1], \
                'ony support 0, 1, 11 for int anchor, ' \
                'but got %i' % anchor

            return anchor

        if isinstance(anchor, numbers.Number):
            assert isinstance(anchor, numbers.Real), \
                'only support float for partition anchor, ' \
                'but got %s' % (type(anchor).__name__)

            return anchor

        assert isinstance(anchor, str), \
            'only support int, float, str for anchor, ' \
            'but got %s' % (type(anchor).__name__)

        if anchor=='both':
            return anchor

        anchors=dict(x='left right'.split(),
                     y='bottom top'.split())[axis]
        assert anchor in anchors, \
            'only allow %s as anchor for axis %s, ' \
            'but got %s' % (str(anchors), axis, anchor)

        return anchors.index(anchor)

    # create axes
    def get_loc_in_root(self, loc='wh'):
        '''
            return location of rect
                with respect to root

            Parameters:
                loc: 'xy' or 'wh'
                    if 'xy', return (x0, y0, x1, y1)
                    if 'wh', return (x0, y0, w,  h )
        '''
        locs=['xy', 'wh']
        assert loc in locs,  \
            'only allow %s for `loc`, ' \
            'but got \'%s\'' % (str(locs), loc)

        manager=self._get_manager()
        root=manager._root_rect

        rootx0=root.left
        rooty0=root.bottom

        rootw=root.width
        rooth=root.height

        axw=self.width
        axh=self.height

        axx0=self.left-rootx0
        axy0=self.bottom-rooty0

        x0=manager.eval_ratio(axx0, rootw)
        y0=manager.eval_ratio(axy0, rooth)
        # assert x0 is not None and y0 is not None

        w=manager.eval_ratio(axw, rootw)
        h=manager.eval_ratio(axh, rooth)
        # assert w is not None and h is not None

        if loc=='xy':
            return x0, y0, x0+w, y0+h

        return x0, y0, w, h

    def create_axes(self, style=None, **kwargs):
        '''
            create axes in rect
            return axes

            Parameters:
                style: None, str
                    frequently used combination of args

                    pass to `params_create_axes`
                        for real kwargs

            Real kwargs:
                tick_params: None or Dict
                    used in `axes.tick_params`

                nlnx, nbnx: bool
                    abbreviation of
                        nlnx: 'not left, then no xlabel'
                        nbnx: 'not bottom, then no ylabel'

                    whether not draw x/ylabels if not left/bottom
        '''
        if style is not None:
            kws=params_create_axes(style)
            for k, v in kws.items():
                assert k not in kwargs, \
                    'conflict arg `%s`' % k

                kwargs[k]=v

        return self._create_axes(**kwargs)

    def _create_axes(self, tick_params=None,
                        nlnx=False, nbny=False,
                        **kwargs):
        '''
            create axes in rect
            return created axes

            basic method to create axes

            Parameters:
                tick_params: None or Dict
                    used in `axes.tick_params`

                nlnx, nbnx: bool
                    abbreviation of
                        nlnx: 'not left, then no xlabel'
                        nbnx: 'not bottom, then no ylabel'

                    whether not draw x/ylabels if not left/bottom
        '''
        if self._axes is not None:
            assert not kwargs, 'axes already created'
            axes=self._axes
        else:
            fig=self._get_manager().create_figure()

            loc=self.get_loc_in_root()
            if any([l is None for l in loc]):
                raise ValueError(f'rect loc indetermined: {loc}')
            axes=fig.add_axes(loc, **kwargs)

            self._axes=axes

        # tick params
        if tick_params:
            axes.tick_params(**tick_params)

        if nlnx and not self.is_left():
            axes.tick_params(labelleft=False)

        if nbny and not self.is_bottom():
            axes.tick_params(labelbottom=False)

        return axes

    def has_axes(self):
        '''
            whether has an axes
        '''
        return self._axes is not None

    def get_axes(self):
        '''
            return axes
        '''
        assert self.has_axes(), \
            'axes not exists'
        return self._axes
    axes=property(get_axes)

    ## x/y-axis share
    def set_axis_share(self, axis, other, ignore_nonexists=False):
        '''
            set to share with other rect or Axes

            Parameters:
                axis: 'x' or 'y'
                    axis to share

                other: Rect, Axes
                    other axes to share
        '''
        check_axis(axis)

        if not self.has_axes():
            if ignore_nonexists:
                return
            else:
                raise Exception('axes not exists')
        axes=self._axes

        if isinstance(other, Rect):
            if not other.has_axes():
                if ignore_nonexists:
                    return
                else:
                    raise Exception('axes not exists in `other`')
            other=other._axes
        elif not isinstance(other, Axes):
            raise TypeError('only allow type `Rect` or `Axes` for `other`')

        # grouper for axis share: `matplotlib.cbook.Grouper`
        grp=getattr(axes, 'get_shared_%s_axes' % axis)()

        ## already shared
        if grp.joined(axes, other):
            return

        getattr(axes, 'share'+axis)(other)

    # to string
    def __repr__(self):
        '''
            representation in string

            return RectGrid(nrows, ncols)[y0:(y1+1), x0:(x0+1)]
                similar as `matplotlib.GridSpec`
        '''
        x0, y0=self._indx0, self._indy0
        x1, y1=self._indx1, self._indy1

        return '%s[%i:%i, %i:%i]' % (repr(self._grid),
                                     y0, y1+1, x0, x1+1)

class Point1D:
    '''
        class for 1d point in grid
    '''
    _AXIS=list('xy')  # allowed axis
    def __init__(self, grid, axis='x', ind=0):
        '''
            init of 1D point

            Parameters:
                grid: RectGrid
                    grid which the point locates at

                axis: 'x' or 'y'
                    axis which the point is along

                ind: int
                    index of the point in grid
        '''
        # grid
        assert isinstance(grid, RectGrid), \
                'only type RectGrid for rect grid, '\
                'but got %s' % (type(grid).__name__)
        self._grid=grid

        # axis
        assert axis in self._AXIS, \
                'unexpected axis: %s' % axis
        self._axis=axis

        # index
        ind=grid._standard_point_index(axis, ind)
        self._ind=ind

    # manager
    def _get_manager(self):
        '''
            return manger of grid

            no register work done
                do it explicitly via RectGrid
            return None if not exists
        '''
        return self._grid._manager

    # as variable
    def _get_var_name(self):
        '''
            return name of the point as a variable
                like 'g0x0',
                    where 'g0' is from grid
                          'x0' is point name

            if no manager at grid, raise Exception
        '''
        pname='%s%i' % (self._axis, self._ind)

        # no name if grid not registered
        return '%s%s' % (self._grid._name, pname)

    ## LnComb
    def to_lncomb(self):
        '''
            return a LnComb instance
        '''
        v=self._get_var_name()
        return LnComb([v], [1], 0)

    ## constraint
    def _set_to(self, p):
        '''
            add constraint to this point
                self = p

            Parameters:
                p: LnComb-support object
                    e.g. other Point
        '''
        self._get_manager()._add_lncomb(self, p)

    ## evaluate
    def _eval(self):
        '''
            evaluate value of the point

            if not determined, return None
        '''
        return self._get_manager()._eval_lncomb(self)

    ## arithmetics
    def __sub__(self, p):
        # self-p
        if isinstance(p, Point1D):
            return p._get_dist_to(self)

        return self.to_lncomb().__sub__(p)

    def __mul__(self, k):
        # self*k
        if k==1:
            return self.__pos__()

        return self.to_lncomb().__mul__(k)

    __neg__=lambda self: self.to_lncomb().__neg__()
    __pos__=lambda self: self

    __add__=lambda self, p: self.to_lncomb().__add__(p)
    __truediv__=lambda self, k: self.__mul__(1/k)

    __rsub__=lambda self, p: self.to_lncomb().__rsub__(p)
    __radd__=lambda self, p: self.to_lncomb().__radd__(p)
    __rmul__=lambda self, k: self.__mul__(k)

    ## user methods
    def set_to(self, p):
        '''
            set point equal to a linear combination, `p`

            `p` should be treated as kind of point,
                meaning to align two points
        '''
        self._set_to(p)

    def eval(self):
        '''
            evaluate variable of this point

            if not determined, return None
        '''
        return self._eval()

    @property
    def name(self):
        '''
            return name as a variable
        '''
        return self._get_var_name()

    # 1d distance
    def _get_dist_to(self, p):
        '''
            return LineSeg1D from self to p
        '''
        return LineSeg1D(self, p)

    # to string
    def __repr__(self):
        '''
            representation in string

            return Point1D(Grid, axis, i)
        '''
        return '%s(%s, %s, %i)' % (type(self).__name__,
                                   repr(self._grid),
                                   repr(self._axis),
                                   self._ind)

class LineSeg1D:
    '''
        class for 1d line segment from p0 to p1, (p0, p1)
            which has a signed distance, p1-p0
    '''
    def __init__(self, p0, p1):
        '''
            line segment from p0 to p1
        '''
        assert p0._axis==p1._axis, \
                'different endpoints for two endpoints'
        assert p0._get_manager() is p1._get_manager(), \
                'different manager for two endpoints'

        self._p0=p0
        self._p1=p1
        self._axis=p0._axis

    # manager
    def _get_manager(self):
        '''
            return manger of p0

            no register work done
                do it explicitly via RectGrid
            return None if not exists

            no check consistence for managers of p0 and p1
        '''
        return self._p0._get_manager()

    # point in line
    def get_partition(self, p):
        '''
            partition in segment
                (1-p)*p0+p*p1
            return LnComb instance

            p=0 for p0, p=1 for p1
            p=0.5 for center
        '''
        return (1-p)*self._p0+p*self._p1

    def get_center(self):
        '''
            center of the segment

            return LnComb instance
        '''
        return self.get_partition(0.5)

    # as variable
    def _get_var_name(self):
        '''
            return name of the dist as a variable
                like 'g0x0_g0x1',
                    where 'g0' is from grid
                          'x0', 'x1' are name of endpoints

            if no manager at grid, raise Exception
        '''
        s0=self._p0._get_var_name()
        s1=self._p1._get_var_name()
        return '%s_%s' % (s0, s1)

    ## LnComb
    def to_lncomb(self):
        '''
            return a LnComb instance of instance
                p1-p0
        '''
        v0=self._p0.to_lncomb()
        v1=self._p1.to_lncomb()

        return v1-v0

    ## constraint
    def _set_to(self, d):
        '''
            add constraint to this distance
                self = d

            Parameters:
                d: LnComb-support object
                    e.g. other Point
        '''
        self._get_manager()._add_lncomb(self, d)

    def _set_ineq(self, d, upper=True):
        '''
            set ineq for the distance
        '''
        self._get_manager()\
            ._add_lncomb_ineq(self, d, upper=upper)

    ## evaluate
    def _eval(self):
        '''
            evaluate value of the distance

            if not determined, return None
        '''
        return self._get_manager()._eval_lncomb(self)

    def _eval_bounds(self):
        '''
            eval bounds
        '''
        return self._get_manager()\
                   ._eval_bounds_of_lncomb(self)

    def _eval_ratio_to(self, d, **kwargs):
        '''
            evaluate ratio to another dist or other LnComb-like,
                self/d
                or k, b for self=k*d+b

            if not determined, return None
        '''
        return self._get_manager()\
                   ._eval_ratio_of_lncombs(self, d, **kwargs)

    ## arithmetics
    def __neg__(self):
        # -self
        return LineSeg1D(self._p1, self._p0)

    def __mul__(self, k):
        # self*k
        if k==1:
            return self.__pos__()

        if k==-1:
            return self.__neg__()

        return self.to_lncomb().__mul__(k)

    __pos__=lambda self: self

    __add__=lambda self, d: self.to_lncomb().__add__(d)
    __sub__=lambda self, d: self.to_lncomb().__sub__(d)
    __truediv__=lambda self, k: self.__mul__(1/k)

    __rsub__=lambda self, d: self.to_lncomb().__rsub__(d)
    __radd__=lambda self, d: self.to_lncomb().__radd__(d)
    __rmul__=lambda self, k: self.__mul__(k)

    ## user methods
    def set_to(self, d):
        '''
            set dist equal to a linear combination, `d`

            `d` should be treated as kind of dist,
                meaning to set two distances equal
        '''
        self._set_to(d)

    def set_bound(self, d, upper=True):
        '''
            set upper or lower bound
        '''
        self._set_ineq(d, upper=upper)

    def set_le(self, d):
        '''
            set self<=d
        '''
        self.set_bound(d, upper=True)

    def set_ge(self, d):
        '''
            set self>=d
        '''
        self.set_bound(d, upper=False)

    def set_lim(self, lim):
        '''
            set lim
        '''
        l, u=lim
        if l is not None:
            self.set_ge(l)
        if u is not None:
            self.set_le(u)

    def eval(self):
        '''
            evaluate this distance
            if not determined, return None
        '''
        return self._eval()

    def eval_bounds(self):
        '''
            eval bounds of this distance
        '''
        return self._eval_bounds()

    def eval_ratio_to(self, d, **kwargs):
        '''
            evaluate ratio to another dist or other LnComb-like,
                self/d or k, b for self=k*d+b
            if not determined, return None
        '''
        return self._eval_ratio_to(d, **kwargs)

    @property
    def name(self):
        '''
            return name as a variable
        '''
        return self._get_var_name()

    # to string
    def __repr__(self):
        '''
            representation in string

            return LineSeg1D(p0, p1)
        '''
        return '%s(%s, %s)' % (type(self).__name__,
                               repr(self._p0),
                               repr(self._p1))

# auxiliary functions
def _parse_ratios(ratios, n):
    '''
        parse argument `ratios` to a list with len `n`
            where `ratios` could be float or list
                if list, first 1 could be omit
                if float `k`, return [1, k, k...]
    '''
    if isinstance(ratios, numbers.Number):
        return [1]+([ratios]*(n-1))

    if len(ratios)==n-1:
        return [1, *ratios]

    assert len(ratios)==n, \
        'only support float or ' \
        'list with %i, %i elements for `ratios`. ' \
        'but got %i' % (n, n-1, len(ratios))

    return ratios

## check function
def check_axis(axis):
    '''
        check axis
    '''
    axs=list('xy')
    assert axis in axs, 'only allow xy for axis'

## special list type
class SetterList(list):
    '''
        a special list with rewrittn `__setitem__`

        element must method `set_to`
            when setitem, `set_to` would be called
    '''
    def __getitem__(self, ind):
        t=super().__getitem__(ind)
        if isinstance(ind, slice):
            t=SetterList(t)
        return t

    def __setitem__(self, ind, v):
        t=self[ind]
        if isinstance(ind, numbers.Integral):
            t.set_to(v)
            return

        if not isinstance(v, abc.Iterable):
            v=[v]*len(t)
        else:
            v=list(v)
            assert len(t)==len(v), \
                'cannot assign %i values to %i elements' \
                    % (len(v), len(t))

        for ti, vi in zip(t, v):
            ti.set_to(vi)

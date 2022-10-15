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
from matplotlib import font_manager

from .linear import LinearManager, LnComb
from .tools import map_to_nested, squeeze_nested, confirm_arg_in, Units
from .tools_class import add_proxy_method
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
            accept grid register

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
    _UNITS.update(['px', 'pixel', 'pts'])

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
        if unit=='pts':
            unit='points'
        elif unit=='px':
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
                size: float, None, or str
                    if str, only support
                        'xx-small', 'x-small', 'small', 'medium', 'large', 
                        'x-large', 'xx-large', 'larger', 'smaller'
                    see `matplotlib.font_manager.font_scalings`
                        for details
        '''
        pts=self.get_size_unit('points')  # points size in inches
        if isinstance(size, numbers.Number):
            return size*pts

        fontsize=plt.rcParams['font.size']
        if size is None:
            return fontsize*pts

        assert isinstance(size, str), \
            'only support float, str or None for fontsize, ' \
            'but got %s' % (type(size).__name__)

        font_scalings=font_manager.font_scalings
        assert size in font_scalings, \
            'only allow fontsize in %s, ' \
            'but got \'%s\'' % (str(list(font_scalings.keys())),
                                size)
        s=font_scalings[size]
        return (s*fontsize)*pts

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
        labsize=self.get_ticksize(axis, 'lab').const
        padsize=max([self.get_ticksize(axis, 'pad', t).const
                        for t in ['major', 'minor']])

        v=nchar*labsize+padsize

        if out:
            ticksize=max([self.get_ticksize(axis, 'tick', t).const
                            for t in ['major', 'minor']])
            v+=ticksize

        return LnComb.lncomb(v)

    ## sep size for axis
    def get_sepsize_axis(self, axis,
                               lab=True, nchar_lab=1,
                               tick_out=True, nchar_tick=1):
        '''
            min separation size for x/y-axis plot
            that is sum of size of tick plot,
                                   axes label,
                                   axes pad
        '''
        s=self.get_ticksize(axis, out=tick_out, nchar=nchar_tick)

        if lab:
            labsize=self.get_fontsize(rcParams['axes.labelsize'])
            labpad=self.get_fontsize(rcParams['axes.labelpad'])

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

    def _eval_ratio_of_lncombs(self, t0, t1):
        '''
            evaluate ratio of two terms, t0/t1

            if not determined, return None
        '''
        return self._vars.eval_ratio_of_lncombs(t0, t1)

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

    def eval_ratio(self, t1, t2):
        '''
            eval t1/t2
        '''
        return self._eval_ratio_of_lncombs(t1, t2)

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

    def eval_wh_ratio(self):
        '''
            evaluate w/h
        '''
        root=self._root_rect
        return self.eval_ratio(root.width, root.height)

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
            if figsize is None:
                figsize=plt.rcParams['figure.figsize']

            w, h=figsize
            assert isinstance(w, numbers.Real) and \
                   isinstance(h, numbers.Real), \
                    'only allow float for `figsize`'

            # ratio w/h eval
            k=self.eval_wh_ratio()
            if k is None:
                root.set_width(w)
                root.set_height(h)
            elif w>k*h:
                root.set_height(h)
            else:
                root.set_width(w)

            # eval from figsize
            w, h=self.eval_figsize()

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
    def _get_rect(self, indx=0, indy=0, xspan=None, yspan=None):
        '''
            return rect with index (indx, indy)

            Parameters:
                indx, indy: int, slice or objects as item of ndarray
                    specify location of rectangle

                xspan, yspan: int, optional
                    span of rect in grid
                    if set to not-None value, indx (or indy) must be int
                    otherwise, just use indx/indy to locate rect
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

        # query buffer of existed rects
        x0, y0=self._standard_rect_xyind(indx=indx, indy=indy)
        x1, y1=self._standard_rect_xyind(indx=x0+xspan-1, indy=y0+yspan-1)

        k=(x0, y0, x1-x0+1, y1-y0+1)
        if k in self._buf_rects:
            return self._buf_rects[k]

        # create new rect
        rect=Rect(self, indx=indx, indy=indy,
                          xspan=xspan, yspan=yspan)
        self._buf_rects[k]=rect

        return rect

    def _get_ith_rect(self, i):
        '''
            return ith rect

            rects is ordered in row-first way
        '''
        indx, indy=self._standard_rect_xyind_of_ith(i)

        return self._get_rect(indx=indx, indy=indy)

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

    def get_rect(self, i):
        '''
            get a rect in Grid

            :param i: int, or tuple (int, int)
                index of rect ot return
                always create new rect

                if int, just return ith rect
                    rects is ordered in row-first way

                if (i, j), means rect in ith row and jth col
        '''
        if isinstance(i, numbers.Number):
            return self._get_ith_rect(i)

        iy, ix=i
        return self._get_rect(indx=ix, indy=iy)

    def get_rects(self, arg='row', reverse=False):
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
        '''
        if isinstance(arg, str):
            return self._get_rects_by_str(arg, reverse=reverse)
        return self._get_rects_by_inds(arg)

    ### auxiliary functions
    def _get_rects_by_str(self, s, reverse=False):
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

        return self._get_rects_by_inds(inds)

    def _get_rects_by_inds(self, inds):
        '''
            fetch rects in given indices
        '''
        return map_to_nested(self.__getitem__, inds,
                    is_scalar=self._is_type_item_index,
                    astype=None)

    ## support syntax: e.g. grid[0, 1], grid[:2, :]
    def _is_type_item_index(self, arg):
        '''
            whether `arg` is valid type for `__getitem__`
        '''
        return isinstance(arg, tuple) or \
               isinstance(arg, slice) or \
               isinstance(arg, numbers.Number)

    def __getitem__(self, prop):
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
            return self._get_ith_rect(prop)
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
        return self._get_rect(indx=indx, indy=indy)

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
    def get_all_widths(self):
        '''
            return list of widths of columns
        '''
        return [self.get_width(i) for i in range(self._nx)]

    def get_all_heights(self):
        '''
            return list of heights of rows
        '''
        return [self.get_height(i) for i in range(self._ny)]

    def get_all_wspaces(self):
        '''
            return list of spaces between columns
        '''
        return [self.get_wspace(i) for i in range(self._nx-1)]

    def get_all_hspaces(self):
        '''
            return list of spaces between rows
        '''
        return [self.get_hspace(i) for i in range(self._ny-1)]

    def get_all_wmargins(self):
        '''
            return list of margins along x-axis
        '''
        if self._parent is None:
            return []
        return [self.get_margin('x', i) for i in range(2)]

    def get_all_hmargins(self):
        '''
            return list of margins along x-axis
        '''
        if self._parent is None:
            return []
        return [self.get_margin('y', i) for i in range(2)]

    ## get collection of dists in group
    _GROUPS_DIST=['width',   'height', 'wspace',  'hspace', 
                 'wmargin', 'hmargin', 'margin', 'sep']
    def get_dists_by_group(self, group='width'):
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
            d0=self.get_dists_by_group('wmargin')
            d1=self.get_dists_by_group('hmargin')
            return [*d0, *d1]

        if group=='sep':
            d0=self.get_dists_by_group('wspace')
            d1=self.get_dists_by_group('hspace')
            return [*d0, *d1]

        return getattr(self, 'get_all_%ss' % group)()

    # set linear constraints
    def set_dists_ratio(self, ratios, dist_group='width'):
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
        dists=self.get_dists_by_group(dist_group)
        if len(dists)<=1:
            return

        self._manager.set_dists_ratio(dists, ratios)

    def set_dists_val(self, vals, dist_group='width'):
        '''
            set value(s) for collection of dists

            Parameters:
                vals: float or list of float
                    value(s) to set

                    if list, must has same len with num of dists

                dist_group: 'width', 'height', 'wspace', 'hspace'
                           'wmargin', 'hmargin', 'margin', 'sep'
                    group of dists to set
        '''
        dists=self.get_dists_by_group(dist_group)
        if not isinstance(vals, abc.Iterable):
            vals=[vals]*len(dists)
        else:
            vals=list(vals)
            assert len(vals)==len(dists), \
                'cannot assign %i vals to %i dists' \
                    % (len(vals), len(dists))

        for di, vi in zip(dists, vals):
            di.set_to(vi)

    def set_dists_bound(self, val, dist_group='sep', upper=True):
        '''
            set dists' bound
        '''
        dists=self.get_dists_by_group(dist_group)
        self._manager.set_dists_bound(val, *dists, upper=upper)

    def set_dists_le(self, maxd, dist_group='width'):
        '''
            set group of dists less-or-equal to val

            :param dist_group: str
                dist group
                    'width', 'height', 'wspace', 'hspace'
                    'wmargin', 'hmargin', 'margin', 'sep'
                see `get_dists_by_group` for detail
        '''
        self.set_dists_bound(maxd, dist_group, upper=True)

    def set_dists_ge(self, mind, dist_group='sep'):
        '''
            set group of dists greater-or-equal to val
            usually used to set not too small separation

            :param dist_group: str
                dist group
                    'width', 'height', 'wspace', 'hspace'
                    'wmargin', 'hmargin', 'margin', 'sep'
                see `get_dists_by_group` for detail
        '''
        self.set_dists_bound(mind, dist_group, upper=False)

    def set_dists_lim(self, lim, dist_group='sep'):
        '''
            set lim for group of dists 

            :param dist_group: str
                dist group
                    'width', 'height', 'wspace', 'hspace'
                    'wmargin', 'hmargin', 'margin', 'sep'
                see `get_dists_by_group` for detail
        '''
        dists=self.get_dists_by_group(dist_group)
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

    def set_rect_sep_margin_ratio(self, ratios, axis='both'):
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
            self.set_rect_sep_margin_ratio(ratios, 'x')
            self.set_rect_sep_margin_ratio(ratios, 'y')
            return

        assert axis in list('xy'), \
                "only support 'x', 'y' for axis"

        # dists
        s=dict(x='width', y='height')[axis]

        dists=[getattr(self, 'get_'+s)(0)]

        n=getattr(self, '_n'+axis)
        if n>1:
            dists.append(getattr(self, 'get_%sspace' % s[0])(0))

        dists.append(self.get_margin(axis, 0))

        # ratios
        ratios=list(_parse_ratios(ratios, 3))
        if len(dists)<3:
            ratios.pop(-2)

        self._manager.set_dists_ratio(dists, ratios)

    def set_seps_zero(self, axis='both'):
        '''
            set separations between rect zero

            Parameters:
                axis: 'x', 'y', or 'both'
                    separations along which axis to set
        '''
        if axis=='both':
            self.set_seps_zero('x')
            self.set_seps_zero('y')
            return

        assert axis in list('xy'), \
                "only support 'x', 'y', 'both' for axis"
        s=dict(x='wspace', y='hspace')[axis]

        self.set_dists_val(0, s)

    def set_seps_ratio_to(self, dist0, ratios, axis='both'):
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
            self.set_seps_ratio_to(dist0, ratios, 'x')
            self.set_seps_ratio_to(dist0, ratios, 'y')
            return

        assert axis in list('xy'), \
                "only support 'x', 'y', 'both' for axis"

        # dist0
        if not isinstance(dist0, LineSeg1D):
            assert isinstance(dist0, RectGrid) or isinstance(dist0, Rect)
            p0, p1=[dist0.get_point(axis, i) for i in [0, -1]]
            dist0=p1-p0

        s=dict(x='wspace', y='hspace')[axis]
        dists=getattr(self, f'get_all_{s}s')()

        if isinstance(ratios, numbers.Number):
            ratios=[ratios]*len(dists)
        else:
            assert len(ratios)==len(dists)

        self._manager.set_dists_ratio([dist0, *dists], [1, *ratios])

    def set_margins_zero(self, axis='both'):
        '''
            set margins to parent rect zero

            Parameters:
                axis: 'x', 'y', or 'both'
                    margins along which axis to set
        '''
        if axis=='both':
            self.set_margins_zero('x')
            self.set_margins_zero('y')
            return

        assert axis in list('xy'), \
                "only support 'x', 'y' for axis"
        s=dict(x='wmargin', y='hmargin')[axis]

        self.set_dists_val(0, s)

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

    def set_loc_at(self, loc, axis='xy', at='parent', locing='wh'):
        '''
            set loc at parent/root rect

            Parameters:
                loc: 4 or 2-tuple of floats
                    fraction at a rect
                    
                    (x0, y0, w, h) or (x0, y0, x1, y1)
                        x0, y0: coordinates for left-bottom corner
                        x1, y1: coordinates for right-top corner
                        w, h: total width and height of grid

                    or just (x0, w) or (x0, x1) for x-axis
                    similar for y-axis

                axis: 'x', 'y', 'xy'
                    which axis to set

                    if 'xy', set both

                at: 'parent', 'root', 'fig', default: 'parent'
                    specify which rect is for the `loc`

                locing: 'wh' or 'xy', default: 'wh'
                    whether `loc` is (x0, y0, w, h) or (x0, y0, x1, y1)
        '''
        # type check
        vs_at=['parent', 'root', 'fig']
        assert at in vs_at, 'ony allow `at` in %s' % str(vs_at)

        vs_lc=['wh', 'xy']
        assert locing in vs_lc, 'ony allow `locing` in %s' % str(vs_lc)

        assert len(loc) in [2, 4] and \
               all([isinstance(t, numbers.Real) for t in loc]), \
                'only allow tuple of 2 or 4 floats as `loc`'

        # locing
        if locing=='wh':
            loc=np.array(loc).reshape(2, -1)
            loc[1]+=loc[0]  # x0+w, y0+h
            loc=list(np.ravel(loc))

        # relative rect
        manager=self.get_manager()

        if at=='parent':
            prect=self.get_parent()
        else:
            prect=manager.rect

        ## axis
        vs_axis=['x', 'y', 'xy']
        assert axis in vs_axis, 'ony allow `axis` in %s' % str(vs_axis)

        self._set_loc_at_by_axis(loc, axis, prect)

    ### auxiliary functions
    def _set_loc_at_by_axis(self, loc, axis, rect0):
        '''
            set loc at `rect0` along axis
            
            Parameters:
                loc: (x0, x1)
                axis: 'x', 'y', or 'xy'
                rect0: relative upper level rect
        '''
        if axis=='xy':
            if len(loc)==2:
                locx=locy=loc
            else:
                locx=(loc[0], loc[2])
                locy=(loc[1], loc[3])

            self._set_loc_at_by_axis(locx, 'x', rect0)
            self._set_loc_at_by_axis(locy, 'y', rect0)

            return

        assert len(loc)==2, \
            'only allow 2 floats for `loc` in axis %s' % axis
        v0, v1=loc

        pp0=rect0.get_point(axis, 0)
        pp1=rect0.get_point(axis, -1)
        pw=pp1-pp0

        p0=self.get_point(axis, 0)
        p1=self.get_point(axis, -1)

        manager=self.get_manager()
        manager._add_lncomb(p0-pp0, v0*pw)
        manager._add_lncomb(p1-pp0, v1*pw)

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
    def create_axes(self, inds=None,
                          sharex=False, sharey=False,
                          omits=None, return_fig=False, **kwargs):
        '''
            create axes in grid

            Parameters:
                inds: None, or arg to `get_rects`
                    if None, use `row`

                sharex, sharey: same type as `matplotlib.pyplot.subplots` or
                                list of collection of indices
                    axes to share x/y-axis

                omits: list of index
                    order of rects to omit, no axes to create

                return_fig: bool
                    whether return fig, axes or just axes
        '''
        if inds is None:
            inds='row'
        rects=self.get_rects(inds)

        # omits
        if omits is None:
            orders_omit=set()
        else:
            orders_omit=set([a.order for a in self.get_rects(omits)])

        # skip omits and construct map {order: rect}
        map_rects={}
        def get_notomit_to_map(a):
            i=a.order
            if i in orders_omit:
                return None
            map_rects[i]=a
            return a

        rects=map_to_nested(get_notomit_to_map, rects,
            is_scalar=lambda a: isinstance(a, Rect),
            skip_none=True)

        # sharex, sharey
        sharex=self._rects_for_axis_share(sharex, map_rects)
        sharey=self._rects_for_axis_share(sharey, map_rects)

        # create axes
        return self._manager.create_axes_in_rects(rects,
                            sharex=sharex, sharey=sharey,
                            return_fig=return_fig, **kwargs)

    def _rects_for_axis_share(self, arg, map_rects):
        '''
            return list of rects collection for axis share

            Parameters:
                arg: arg for axis share
                    2 types:
                        - same as `matplotlib.pyplot.subplots`
                        - args passed to `get_rects`

                map_rects: dict {order: rects}
        '''
        if arg is None:
            return None

        if isinstance(arg, bool):
            arg='all' if arg else 'none'

        if arg=='none':
            return None

        rects=self.get_rects(arg)
        if not rects:
            return None

        # for squeeze array, like arg='all'
        if isinstance(rects[0], Rect):
            rects=[rects]

        def get_rect_from_map(a):
            i=a.order
            return map_rects.get(i, None)

        return map_to_nested(get_rect_from_map, rects,
            is_scalar=lambda a: isinstance(a, Rect),
            skip_none=True)

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

    ## getattr: rect.x0, rect.x1, rect.y0, rect.y1
    def __getattr__(self, prop):
        if prop[0] in ['x', 'y'] and prop[1:].isdigit():
            a, i=prop[0], int(prop[1:])
            return self.get_point(a, i)

        raise AttributeError('unexpected attr: %s' % prop)

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

    def _eval_ratio_to(self, d):
        '''
            evaluate ratio to another dist or other LnComb-like,
                self/d

            if not determined, return None
        '''
        return self._get_manager()\
                   ._eval_ratio_of_lncombs(self, d)

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

    def eval_ratio_to(self, d):
        '''
            evaluate ratio to another dist or other LnComb-like,
                self/d
            if not determined, return None
        '''
        return self._eval_ratio_to(d)

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

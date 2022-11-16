#!/usr/bin/env python3

'''
    functions to add patches, like circle, rectangle
'''

import numbers

import numpy as np

import matplotlib.patches as mpatches
import matplotlib.path as mpath

from .path import ClosedPath

_patches={}
__all__=['add_patch']

def _register_padder(f):
    name=f.__name__
    pname=name.split('_', maxsplit=1)[1].replace('_', ' ')

    def f1(*args, fill=True, marker=None, kws_marker={}, **kwargs):
        # fill
        kwargs['fill']=bool(fill)
        if fill and isinstance(fill, numbers.Number):
            if 'alpha' not in kwargs:
                kwargs['alpha']=fill

        # marker of xy0
        if marker is not None:
            ax, (x0, y0)=args[:2]
            if 'color' in kwargs and 'color' not in kws_marker:
                kws_marker['color']=kwargs['color']
            ax.scatter([x0], [y0], marker=marker, **kws_marker)

        return f(*args, **kwargs)
    f1.__doc__=f.__doc__

    _patches[pname]=f1
    __all__.append(name)

    return f1

# circle
@_register_padder
def add_circle(ax, xy0, r, **kwargs):
    circle=mpatches.Circle(xy0, r, **kwargs)
    return ax.add_patch(circle)

# ellipse
@_register_padder
def add_ellipse(ax, xy0, w, h, angle=0, **kwargs):
    ellipse=mpatches.Ellipse(xy0, w, h, angle=angle, **kwargs)
    return ax.add_patch(ellipse)

# rectangle
@_register_padder
def add_rect(ax, xy0, w, h, angle=0, **kwargs):
    rect=mpatches.Rectangle(xy0, w, h, angle=angle, **kwargs)
    return ax.add_patch(rect)

# polygon
@_register_padder
def add_regular_polygon(ax, xy0, n, r, orientation=0, **kwargs):
    '''
        regular polygon
            which has vertices distributed along a cirle uniformly
                with first vertex in right up direction
    '''
    pgon=mpatches.RegularPolygon(xy0, n, r, orientation=orientation, **kwargs)
    return ax.add_patch(pgon)

@_register_padder
def add_polygon(ax, *args, pathby='vert', **kwargs):
    '''
        polygon specified by vertices

        :param path: 'vert', 'angle' or 'polar'
            how the args specify polygon path

            if 'vert': `args` for list of xys
            if 'angle': `xy0`, `angles`, `lens`
            if 'polar': `xy0`, `angles`, `radius`
    '''
    if pathby=='vert':
        path=path_polygon_by_verts(*args)
    elif pathby=='angle':
        kws={}
        for k in ['orientation', 'lens']:
            if k in kwargs:
                kws[k]=kwargs.pop(k)
        path=path_polygon_by_angles(*args, **kws)
    elif pathby=='polar':
        kws={}
        for k in ['orientation', 'radius']:
            if k in kwargs:
                kws[k]=kwargs.pop(k)
        path=path_polygon_by_polar(*args, **kws)
    else:
        raise ValueError(f'unknown path construct: {pathby}')

    patch=mpatches.PathPatch(path, **kwargs)
    return ax.add_patch(patch)

## path of polygon
def path_polygon_by_verts(*xys):
    '''
        polygon path from verticies
    '''
    n=len(xys)
    assert n>=3, 'at least 3 vertices'

    return ClosedPath(xys, closed=True)

def path_polygon_by_polar(xy0, angles, radius=1, orientation=0):
    '''
        polygon path by polar coordinates

        :param angles: int, list of float
            angles in degree relative to base direction
                that is y-axis if `orientation==0`

            if int, mean number of vertices

        :param radius: float, list of float or callable
            if float, same radius for all vertices
            if callable, yield real radius by feeding `angles`
    '''
    if isinstance(angles, numbers.Number):
        angles=np.linspace(0, 360, angles, endpoint=False)
    else:
        angles=np.asarray(angles)

    n=len(angles)
    assert n>=3, 'at least 3 angles specified'

    thetas=(angles+orientation)*np.pi/180

    # radius
    if isinstance(radius, numbers.Number):
        radius=[radius]*n
    elif callable(radius):
        radius=radius(angles)
    radius=np.asarray(radius)
    assert len(radius)==n, 'mismatch len between `angles` and `radius`'

    # xy
    x0, y0=xy0
    xs=x0-radius*np.sin(thetas)
    ys=y0+radius*np.cos(thetas)

    return path_polygon_by_verts(*zip(xs, ys))

def path_polygon_by_angles(xy0, angles, lens=1, orientation=None):
    '''
        polygon path by angles between edges

        :param angles: int or list of float
            angles between edges
            first one is relative to x-axis

            if int, it is number of vertices
                use same angle for all
                    that is 360/n

        :param orientation: None or float
            orientation of first edge

            if None, it's given by 0th element in `angles`

        :param lens: float or list of float
            lengths of edges

            if float, all edges has same length
    '''
    if isinstance(angles, numbers.Number):
        angles=[360/angles]*(angles-2)
        if orientation is None:
            orientation=0
        angles=[orientation, *angles]
    elif orientation is not None:
        angles=[orientation, *angles]

    n=len(angles)
    assert n>=2, 'at least 2 angles specified'

    thetas=np.cumsum(angles)*np.pi/180

    # lengths
    if isinstance(lens, numbers.Number):
        lens=[lens]*n
    elif len(lens)!=n:
        raise ValueError('num of `lens` mismatch with that of `angles`')
    lens=np.asarray(lens)

    # vector of edges
    dxs=lens*np.cos(thetas)
    dys=lens*np.sin(thetas)

    x0, y0=xy0
    xs=np.cumsum([x0, *dxs])
    ys=np.cumsum([y0, *dys])

    return path_polygon_by_verts(*zip(xs, ys))

# path
@_register_padder
def add_path(ax, *verts, codes=None, closed=False, **kwargs):
    '''
        add path by verts

        :param codes: str, path code or list of code
            codes to create path

            a special code, str 'bezier'
                used to create bezier curve

            other scalar code: LINETO, CURVE3, CURVE4
    '''
    pass

# method to call patch drawer
def add_patch(ax, patch, *args, **kwargs):
    patch=patch.lower()
    if patch not in _patches:
        raise ValueError(f'only allow add patch for {list(_patches.keys())}, '
                         f'but got {patch}')

    func=_patches[patch]
    return func(ax, *args, **kwargs)

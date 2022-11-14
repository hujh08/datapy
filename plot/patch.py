#!/usr/bin/env python3

'''
    functions to add patches, like circle, rectangle
'''

import numbers

import numpy as np

import matplotlib.patches as mpatches
import matplotlib.path as mpath

from .colors import get_next_color_in_cycle

_patches={}
__all__=['add_patch']

def _register_padder(f):
    name=f.__name__
    pname=name.split('_', maxsplit=1)[1].replace('_', ' ')

    def f1(ax, *args, fill=True, **kwargs):
        if (not fill) and \
           ('fc' not in kwargs and 'facecolor' not in kwargs):
            kwargs['fc']='none'
            if 'color' not in kwargs and \
               'ec' not in kwargs and \
               'edgecolor' not in kwargs:
                kwargs['ec']=get_next_color_in_cycle(ax)
        elif isinstance(fill, numbers.Number) and 'alpha' not in kwargs:
            kwargs['alpha']=fill
        return f(ax, *args, **kwargs)
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
def add_regular_polygon(ax, xy0, n, r, angle=0, **kwargs):
    '''
        regular polygon
            which has vertices distributed along a cirle uniformly
                with first vertex in right up direction
    '''
    pgon=mpatches.RegularPolygon(xy0, n, r, orientation=angle, **kwargs)
    return ax.add_patch(pgon)

@_register_padder
def add_polygon(ax, *args, pathby='vert', **kwargs):
    '''
        polygon specified by vertices

        :param path: 'vert' or 'angle'
            how the args specify polygon path

            if 'vert': `args` for list of xys
            if 'angle': `xy0`, `angles`, `lens`
    '''
    if pathby=='vert':
        path=path_polygon_by_verts(*args)
    elif pathby=='angle':
        kws={}
        if 'lens' in kwargs:
            kws['lens']=kwargs.pop('lens')
        path=path_polygon_by_angles(*args, **kws)
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

    verts=[*xys, xys[0]]

    codes=[mpath.Path.LINETO]*(n-1)
    codes=[mpath.Path.MOVETO, *codes, mpath.Path.CLOSEPOLY]
    
    return mpath.Path(verts, codes)

def path_polygon_by_angles(xy0, angles, lens=1):
    '''
        polygon path by angles between edges

        :param angles: int or list of float
            angles between edges
            first one is relative to x-axis

            if int, it is number of vertices
                use same angle for all
                    that is 360/n

        :param lens: float or list of float
            lengths of edges

            if float, all edges has same length
    '''
    if isinstance(angles, numbers.Number):
        angles=[360/angles]*(angles-2)
        angles=[0, *angles]
    n=len(angles)
    assert n>=2, 'at least 2 angles specified'

    thetas=np.cumsum(angles)*np.pi/180

    if isinstance(lens, numbers.Number):
        lens=[lens]*len(thetas)
    elif len(lens)!=len(thetas):
        raise ValueError('num of `lens` mismatch with that of `angles`')
    lens=np.asarray(lens)

    dxs=lens*np.cos(thetas)
    dys=lens*np.sin(thetas)

    x0, y0=xy0
    xs=np.cumsum([x0, *dxs])
    ys=np.cumsum([y0, *dys])

    return path_polygon_by_verts(*zip(xs, ys))

# method to call patch drawer
def add_patch(ax, patch, *args, **kwargs):
    patch=patch.lower()
    if patch not in _patches:
        raise ValueError(f'only allow add patch for {list(_patches.keys())}, '
                         f'but got {patch}')

    func=_patches[patch]
    return func(ax, *args, **kwargs)

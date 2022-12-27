#!/usr/bin/env python3

'''
    functions to handle color in matplotlib
'''

import numbers

import numpy as np

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

__all__=['get_next_color',
         'cmap_from_colors', 'cmap_from_cpair']

# color cycle
def get_next_color_in_cycle(ax):
    '''
        get next color in color cycle of an axes
    '''
    ltmp,=ax.plot([])
    color=ltmp.get_color()
    ltmp.remove()

    return color

get_next_color=get_next_color_in_cycle

def get_color_cycler(*colors):
    '''
        get color cycler

        Use it like:
            cyc=get_color_cycler(*'rgb')
            for data, c in zip(datas, cyc):
                plt.plot(*data, ..., color=c)

        OR
            itercyc=iter(cyc)
            c=next(itercyc)
            plt.plot(..., color=c)

        Unlike `plt.cycler`, which should be used like
            cyc=plt.cycler(color='rgb')
            for data, t in zip(datas, cyc()):  # must call it for infinite cycle
                plt.plot(*data, ..., color=t['color'])
        For more complicated style, not just color
            `plt.cycler` is better choice

        if no colors given, use `plt.rcParams['axes.prop_cycle']`
    '''
    import itertools

    if len(colors)==0:
        prop_cycle=plt.rcParams['axes.prop_cycle']
        colors=prop_cycle.by_key()['color']

    return itertools.cycle(colors)

# color map
def cmap_from_anchors(rs, gs, bs, alphas=None,
                        rinds=None, ginds=None, binds=None, ainds=None,
                        name='_cmap_from_anchors', **kwargs):
    '''
        create color map from given anchors in RGB(or RGBa) individually

        Parameters:
            rs, gs, bs: float, list of values
                values of RGB in anchor

                if float, same value for all anchors

                value could be float or pair (yleft, yright)
                    if pair, it is for discontinuities

            alphas: None, float, or list of values
                values of alpha in anchor
                    float or pair for each value,
                    similar as `rs`, `gs`, `bs`

            rinds, ginds, binds, ainds: list of float
                position of anchors

                if None, use equal spacing, `np.linspace(0, 1, n)`
    '''
    rgbas=[(rs, rinds), (gs, ginds), (bs, binds)]
    if alphas is not None:
        rgbas.append((alphas, ainds))

    # create dict for segment
    cdict={}
    keys=['red', 'green', 'blue', 'alpha']
    for k, (vals, anchs) in zip(keys, rgbas):
        # standardize anchors and its values
        if isinstance(vals, numbers.Number):
            n=2 if anchs is None else len(anchs)
            vals=[vals]*n
        n=len(vals)

        if anchs is None:
            anchs=np.linspace(0, 1, n)
        else:
            assert len(anchs)==n and \
                   anchs[0]==0 and anchs[-1]==1 and \
                   (np.diff(anchs)>=0).all()

        rows=[]
        for xi, ys in zip(anchs, vals):
            if isinstance(ys, numbers.Number):
                y0=y1=ys
            else:
                y0, y1=ys

            assert 0<=xi<=1 and 0<=y0<=1 and 0<=y1<=1
            rows.append([xi, y0, y1])

        cdict[k]=rows

    return LinearSegmentedColormap(name, segmentdata=cdict, **kwargs)

def cmap_from_rgb_anchors(*colors, alphas=None, cinds=None, ainds=None,
                            name='_cmap_from_rgb_anchors', **kwargs):
    '''
        create color map for anchors
            which is given by RGB (together) and alpha

        NOTE: discontinuity not supportted in this cmap

        Parameters:
            args `colors`: color str, (color, alpha), or (r, g, b, a) for each arg
                RGB value in an anchor

                if only one arg given, same RGB for all anchor

            alphas: None, float, or list of float or pair
                values of alpha in anchor

                if not None, alpha in `colors` would be ignored
                otherwise, alpha in `colors` would be used
                    and then `ainds` would be ignored and use `cinds` as `ainds`

            cinds, ainds: list of float
                position of anchors for RGB and alpha

                if None, use equal spacing, `np.linspace(0, 1, n)`

    '''
    rgbas=[[] for _ in range(4)]
    anyrgba=False
    for c in colors:
        args=(c,)
        if not isinstance(c, str):
            if len(c)==2:
                args=tuple(c)
                anyrgba=True
            anyrgba = anyrgba or (len(c)==4)

        for i, v in enumerate(mcolors.to_rgba(*args)):
            rgbas[i].append(v)

    if len(colors)==1:
        rgbas=[t[0] for t in rgbas]  # use constant if only one args given

    # None alphas
    if alphas is None and anyrgba:
        alphas=rgbas[-1]
        ainds=cinds

    return cmap_from_anchors(*rgbas[:-1], alphas=alphas,
                                rinds=cinds, ginds=cinds, binds=cinds, ainds=ainds,
                                name=name, **kwargs)

def cmap_from_colors(*colors, nodes=None,
                        name='_cmap_from_colors', alpha=None, **kwargs):
    '''
        segemented color map with equal spacings from list of colors

        Parameters:
            colors: color str, rgb or rgba array

            nodes: None, or list of floats within [0, 1]
                corresponding node of given colors

                if None, use default `np.linspace(0, 1, n)`
                    where `n = len(colors)`

            name: str, optional, default: '_cmap_from_colors'
                optional cmap name

            alpha: None, or float
                alpha of colors

                if not None, use same `alpha` for all
    '''
    if alpha is not None:
        assert isinstance(alpha, numbers.Number)
        colors=[mcolors.to_rgba(c, alpha) for c in colors]

    if nodes is not None:
        assert len(nodes)==len(colors) and all([0<=i<=1 for i in nodes])
        colors=list(zip(nodes, colors))

    return LinearSegmentedColormap.from_list(name, colors, **kwargs)

def cmap_from_cpair(color1, color2, name='_cmap_from_cpair', alpha=None, **kwargs):
    '''
        create color map moving from `color1` to `color2` linearly

        Parameters:
            color1, color2: color str, rgb or rgba array
                colors correspond to 0., 1.

            name: str, optional, default: '_cmap_from_cpair'
                optional cmap name

            alpha: None or float
                alpha of colors

                if not None, use same `alpha` for all
    '''
    return cmap_from_colors(color1, color2, name=name, alpha=alpha, **kwargs)

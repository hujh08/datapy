#!/usr/bin/env python3

'''
    useful functions for plot task
'''

import numbers

import numpy as np

# filter out data

def get_lim_oneside_of(xs, xlim=None, limtype='v', left=True):
    '''
        leftmost/rightmost of data
    '''
    lts_valid=['value', 'v', 'quant', 'q']
    assert limtype in lts_valid, \
           f'only allow `limtype` in {lts_valid}, but got "{limtype}"'

    if xlim is None:
        return np.min(xs) if left else np.max(xs)
    if limtype in ['q', 'quant']:
        return np.quantile(xs, xlim)
    return xlim

def get_lims_of(xs, xlim=None, limtype='v'):
    '''
        get limit of data

        :param limtype: str 'value', 'v', 'quant', 'q', or pair
            how to use the given xlim
    '''

    if isinstance(limtype, str):
        lt0=lt1=limtype
    else:
        lt0, lt1=limtype

    if xlim is None:
        x0=x1=None
    else:
        x0, x1=xlim

    x0=get_lim_oneside_of(xs, x0, lt0)
    x1=get_lim_oneside_of(xs, x1, lt1, left=False)

    return x0, x1

def filter_by_lim(xs, *others, xlim=None, limtype='value'):
    '''
        filter data by lim
        support multi-dimension data

        Parameters:
            xs: 1d array
                basic array, where the limit is applied

            others: 1d arrays
                same shape with xs

            xlim: [x0, x1]
                limit to apply

            limtype: str 'value', 'v', 'quant', 'q', or pair
                how to use the given xlim
    '''
    if isinstance(limtype, str):
        limtype=[limtype]*2

    xs=np.asanyarray(xs)
    others=list(map(np.asanyarray, others))

    # limit cut
    if xlim is not None:
        x0, x1=xlim
        lt0, lt1=limtype

        m=np.ones(xs.shape, dtype=bool)

        if x0 is not None:
            x0=get_lim_oneside_of(xs, x0, lt0)
            m=np.logical_and(m, xs>=x0)

        if x1 is not None:
            x1=get_lim_oneside_of(xs, x1, lt1, left=False)
            m=np.logical_and(m, xs<=x1)

        xs=xs[m]
        others=[ys[m] for ys in others]

    if not others:
        return xs
    return xs, *others

# stats

## density map
def calc_density_map_2d(xs, ys, bins=None,
                            xlim=None, ylim=None, limtype='v',
                            kde=False):
    '''
        calculate density map of 2d dataset

        Paramters:
            xs, ys: 1d array with same length
                dataset

            bins: None, int, array, or pair [xbins, ybins]
                same as np.histogram2d

                if None, use 20, or 100 (for kde)
                if int or array, same setup for both xbins and ybins

                for xbins/ybins, 2 cases:
                    - int: number of bins
                        region from xlim/ylim
                    - array: xedges/yedges

                NOTE: to distinguish array and [xbins, ybins]
                    any iterable input with len = 2 would be treated as [xbins, ybins]

            xlim/ylim: None, pair [x0, x1]/[y0, y1]
                data region to calculate

                if None or None in pair, use min or max as default

                it works only when bins edges are not given explicitly

            limtype: str {'value', 'v', 'quant', 'q'}, pair of str, or [xlimtype, ylimtype]
                use xlim/ylim as absolute value or quantile

                if str or pair of str, xlimtype=ylimtype=limtype
                    for pair of str, elements is type for x0, x1 respectively

                a special case is [[str0, str1]]
                    means xlimtype=st0, ylimtype=str1
    '''
    # bins split
    if bins is None:
        xbins=ybins=100 if kde else 20
    elif isinstance(bins, numbers.Number) or len(bins)!=2:
        xbins=ybins=bins
    else:
        xbins, ybins=bins

    ## limtype
    if isinstance(limtype, str) or all([isinstance(t, str) for t in limtype]):
        xlt=ylt=limtype
    elif len(limtype)==1:  # special case [[str0, str1]]
        assert all([isinstance(t, str) for t in limtype[0]])
        xlt, ylt=limtype[0]
    else:
        xlt, ylt=limtype

    xedges=split_bins_for_1d(xs, xbins, xlim, xlt)
    yedges=split_bins_for_1d(ys, ybins, ylim, ylt)

    xcents=(xedges[:-1]+xedges[1:])/2
    ycents=(yedges[:-1]+yedges[1:])/2

    # filter data in region
    xs, ys=filter_by_lim(xs, ys, xlim=[xedges[0], xedges[-1]])
    ys, xs=filter_by_lim(ys, xs, xlim=[yedges[0], yedges[-1]])

    # kde
    if kde:
        return _densmap_by_kde(xs, ys, xcents, ycents)

    # count in bins
    cnts, *_=np.histogram2d(xs, ys, bins=[xedges, yedges])
    cnts=cnts.T   # shape of result from np.histogram2d is (nx, ny)

    # convert to density map
    dxs=np.diff(xedges).reshape(1, -1)
    dys=np.diff(yedges).reshape(-1, 1)
    dxys=dxs*dys
    dens=(cnts/len(xs))/dxys

    return dens, (xcents, ycents)

def quants_to_levels(weights, quants):
    '''
        list of quantiles to levels in density map

        `quantile` means fraction of points outside of countour
    '''
    
    quants=np.asarray(quants)

    # sort weights
    weights=np.ravel(weights)
    assert np.all(weights>=0)

    weights=np.sort(weights)[::-1]
    ws_cum=np.cumsum(weights)/np.sum(weights)

    # compute levels
    levels=np.interp(1-quants, ws_cum, weights)

    return levels

## auxiliary functions
def _densmap_by_kde(xs, ys, xcents, ycents):
    # density by KDE

    from scipy.stats import gaussian_kde

    # meshgrid to compute
    pxx, pyy=np.meshgrid(xcents, ycents, indexing='xy')
    pxy=np.row_stack([pxx.ravel(), pyy.ravel()])

    # kde
    datas=np.row_stack([xs, ys])
    fkde=gaussian_kde(datas)
    dens=np.reshape(fkde(pxy).T, pxx.shape)

    return dens, (xcents, ycents)

def split_bins_for_1d(xs, bins, xlim=None, limtype='v'):
    '''
        split bins for 1d data

        return: array of bins' edges

        :param bins: int or array
            if array, bins edges
                nothing would be done
            if int, number of bins
    '''
    # edges explicitly given
    if not isinstance(bins, numbers.Number):
        return bins

    # number of bins given
    nbin=bins
    x0, x1=get_lims_of(xs, xlim, limtype=limtype)

    return np.linspace(x0, x1, nbin+1)
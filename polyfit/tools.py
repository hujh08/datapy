#!/usr/bin/env python3

'''
    simple tools for polynomial fit
'''

import numbers

import numpy as np

from .poly import NDPoly
from ._tools import check_poly_ncoeffs_dim_deg

__all__=['polyfit_1d',
         'polyfit_ndim', 'factory_ndpoly']

# 1-dimension poly
def polyfit_1d(ys, xs, deg, monic=False, ret='all', **kwargs):
    '''
        1-dimension polynomial fit

        Fit a polynomial `y = f(x) = p[0] * x**deg + ... + p[deg]`

        Parameters:
            xs, ys, deg, **kwargs: args for fit
                see `numpy.polyfit` for detail

                optional `kwargs`:
                    rcond, w: for how to fit
                    full, cov: for what is returned from `numpy.polyfit`

            monic: bool, default False
                whether to fit with monic poly, that is p[0] fixed to 1

                it works only when deg>=1

            ret: str 'fit', 'coeff', 'poly', 'all', default 'poly'
                what to return for this function

                if 'fit', return result of `numpy.polyfit`
                    controlled by args `full` and `cov`

                    if `monic` True, first coefficient 1 would not return

                if 'coeff', only return coefficients

                if 'poly', return class `numpy.poly1d`
                    which is also callable as the result poly function

                if 'all', return both poly class and all fit result
                    (poly class, fit result)
    '''
    if ret not in ['fit', 'coeff', 'poly', 'all']:
        raise ValueError(f'unexpected value for arg `ret`: {ret}')

    # monic fit
    if monic:
        if deg<1:
            raise ValueError('only support monic poly fit for `deg >= 1`')

        xs=np.asarray(xs)
        ys=np.asarray(ys)
        ys=ys-xs**deg
        deg=deg-1

    fitres=np.polyfit(xs, ys, deg=deg, **kwargs)

    if ret=='fit':
        return fitres

    if type(fitres) is tuple:
        coeffs=fitres[0]
    else:
        coeffs=fitres

    if ret=='coeff':
        return coeffs

    # to 'numpy.poly1d' class
    if monic:
        coeffs=[1, *coeffs]

    p=np.poly1d(coeffs)

    if ret=='poly':
        return p

    return p, fitres

# multi-dimensional poly
def polyfit_ndim(ys, *xs, degs=1, rcond=None, ret='all'):
    '''
        multi-dimensional polynomial fit

        return coeffs or callable function
            coeffs: list of coefficient  for terms in lexicographic order

            function: call with form `f(x1, x2, ..., xn)`
                where each `xi` has same shape

        Parameters:
            ret: str 'fit', 'coeff', 'poly', 'all', default 'all'
                what to return for this function

                if 'fit', return result of `numpy.linalg.lstsq`
                    controlled by args `full` and `cov`

                    if `monic` True, first coefficient 1 would not return

                if 'coeff', only return coefficients

                if 'poly', return `NDPoly` instance
                    called with from `p(x1, x2, ..., xn)`

                if 'all', return function and fit result
                    (func, result of fit)
    '''
    dim=len(xs)
    if dim<0:
        raise ValueError('no `xs` provided for fitting')

    if ret not in ['fit', 'coeff', 'func', 'all']:
        raise ValueError(f'unexpected value for arg `ret`: {ret}')

    # NDPoly instance
    p=NDPoly(degs, dim=dim)
    fitres=p.fit(ys, *xs, rcond=rcond, inplace=True)
    
    if ret=='fit':
        return fitres

    coeffs=np.asarray(p.coeffs)
    if ret=='coeff':
        return coeffs

    if ret=='poly':
        return p

    return p, fitres

def factory_ndpoly(coeffs, dim=None, deg=None, **kwargs):
    '''
        factory for NDPoly instance
            with given `coeffs`, `dim` and `deg`

        :param deg: None or int
            poly degree

        :param coeffs: list of number
            coefficients of poly terms in lexicographic order

        optional kwargs: order, descending
            used to specify how `coeffs` is arranged
                like in lexicographic order, `order='lex'`
                     or degreee-lexicographic order, `order='deglex'`

        if `dim` or `deg` is None, to guess them by len of coeffs
    '''
    ncoeffs=len(coeffs)
    dim, deg=check_poly_ncoeffs_dim_deg(ncoeffs, dim=dim, deg=deg)

    p=NDPoly(deg, coeffs=coeffs, dim=dim)
    return p

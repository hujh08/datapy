#!/usr/bin/env python3

'''
    simple tools for polynomial fit
'''

import numbers

import numpy as np

__all__=['polyfit_1d',
         'polyfit_ndim', 'factory_func_poly_nd']

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
def polyfit_ndim(ys, *xs, deg=1, rcond=None, ret='all'):
    '''
        multi-dimensional polynomial fit

        return coeffs or callable function
            coeffs: list of coefficient  for terms in lexicographic order

            function: call with form `f(x1, x2, ..., xn)`
                where each `xi` has same shape

        Parameters:
            ret: str 'fit', 'coeff', 'func', 'all', default 'all'
                what to return for this function

                if 'fit', return result of `numpy.linalg.lstsq`
                    controlled by args `full` and `cov`

                    if `monic` True, first coefficient 1 would not return

                if 'coeff', only return coefficients

                if 'func', return function for poly
                    called with from `f(x1, x2, ..., xn)`

                if 'all', return function and fit result
                    (func, result of fit)
    '''
    dim=len(xs)
    if dim<0:
        raise ValueError('no `xs` provided for fitting')

    if ret not in ['fit', 'coeff', 'func', 'all']:
        raise ValueError(f'unexpected value for arg `ret`: {ret}')

    # align shape of `xs`, `ys`
    ys, *xs=_align_arrs_shape((ys, *xs))

    arr_xs=np.column_stack([np.ravel(x) for x in xs])  # to 2d array (N, dim)
    terms=_poly_terms_ndim(arr_xs, deg)
    arrs=np.column_stack(terms)

    fitres=np.linalg.lstsq(arrs, ys, rcond=rcond)

    if ret=='fit':
        return fitres

    coeffs=fitres[0]
    if ret=='coeff':
        return coeffs

    # polynomial function
    func_poly_nd=factory_func_poly_nd(coeffs, dim=dim, deg=deg)

    if ret=='func':
        return func_poly_nd

    return func_poly_nd, fitres

def factory_func_poly_nd(coeffs, dim=None, deg=None):
    '''
        factory for func of poly
            with given `coeffs`, `dim` and `deg`

        :param coeffs: list of number
            coefficients of poly terms in lexicographic order

        if `dim` or `deg` is None, to guess them by len of coeffs
    '''
    ncoeffs=len(coeffs)
    dim, deg=_check_poly_ncoeffs_dim_deg(ncoeffs, dim=dim, deg=deg)

    # constant poly
    if ncoeffs==1:
        c=coeffs[0]
        func_poly_nd=lambda *xs: c
        func_poly_nd.__doc__='constant poly function'

        return func_poly_nd
    
    # poly function
    def func_poly_nd(*xs):
        if len(xs)!=dim:
            s=f'dims of `xs` and poly not match: {len(xs)}!={dim}'
            raise ValueError(s)

        # align ndarray's shape
        xs=_align_arrs_shape(xs)
        shape=xs[0].shape

        arr_xs=np.column_stack([np.ravel(x) for x in xs])
        terms=_poly_terms_ndim(arr_xs, deg)

        res=coeffs[0]*terms[0]
        for k, t in zip(coeffs[1:], terms[1:]):
            res=res+k*t

        if not shape:  # scalar
            res=res[0]
        else:
            res=res.reshape(shape)

        return res

    func_poly_nd.__doc__=f'''
        func for poly with dim {dim}, deg {deg}

        :params xs: number or nd array
            same shape for nd array
    '''

    return func_poly_nd

## auxiliary funcitons
def _align_arrs_shape(xs):
    '''
        align arrays' shape

        :param xs: sequence of number or ndarray
            must have same shape for all nd array
                raise ValueError if not

            if mixed scalar and ndarray,
                fill `scalar` to common shape of ndarray
    '''
    n_xs=len(xs)
    xs=[np.asarray(x) for x in xs]

    is_scalars=[x.ndim==0 for x in xs]
    n_scalars=sum(is_scalars)
    if n_scalars is n_xs:
        return xs

    # ndarray in `xs`
    arr_xs=[x for k, x in zip(is_scalars, xs) if not k]

    shape=arr_xs[0].shape
    if any([a.shape!=shape for a in arr_xs[1:]]):
        raise ValueError('different shape for ndarrays in `xs`')

    ## convert number to array
    if n_scalars!=0:
        a0=arr_xs[0]
        xs=[np.full_like(a0, x, dtype=x.dtype) if k else x
                for k, x in zip(is_scalars, xs)]

    return xs

### num of poly terms
def _check_poly_ncoeffs_dim_deg(ncoeffs, dim=None, deg=None):
    '''
        check consistency for num of coeffs `ncoeffs` and `dim`, `deg`
        if `dim` or `deg` unknown, find it

        return (dim, deg)
    '''
    # value check
    if ncoeffs<=0:
        raise ValueError('empty `coeffs` given')

    for s in ['dim', 'deg']:
        v=locals()[s]
        if v is not None and v<0:  # only allow non-negative
            raise ValueError(f'only allow `{s}` >= 0')

    # both dim and deg None: undetermined
    if dim is None and deg is None:
        # (dim, deg)=(n-1, 1) or (1, n-1) has same num of terms
        raise ValueError('cannot determine `dim` and `deg` if both None')

    # Neither None: check consistency
    if dim is not None and deg is not None:
        numt=_num_terms_of_poly_nd(dim, deg)
        if ncoeffs!=numt:
            raise ValueError('num of `coeffs` not match with that of poly terms')

        return dim, deg

    # one is not None:
    if dim is not None:  # deg None
        deg=_find_unknown_for_nterms(ncoeffs, dim, unknown='deg')
    else:  # deg not None, dim None
        dim=_find_unknown_for_nterms(ncoeffs, deg, unknown='dim')

    return dim, deg

def _find_unknown_for_nterms(nterms, dim, unknown='deg'):
    '''
        find unknown `deg` or `dim` for `nterms`
            nterms: number of terms
            deg: poly degree
            dim: dimension, number of variables in poly

        Since N(g, m) is symmetric (N(g,m)=N(m,g))
            it works for both `dim` and `deg`

        if not exists, raise ValueError

        :param unknown: str
            name for the unknown `deg` or `dim`

        :param dim: `deg` or `dim`, use dim as example
            Since N(g,m)=N(g,m), same for deg
    '''
    if dim==0:
        if nterms!=1:
            s=f'only 1 term for 0 `{unknown}`, but got {nterms}'
            raise ValueError(s)
        return None  # could be any deg

    if dim==1:
        return nterms-1

    # bound of deg: N(g, m) >= (g/m)^m, g+1
    maxdeg=min(nterms-1, int(nterms**(1/dim)*dim))

    # array of number of terms
    arr_nums=_num_terms_of_poly_nd(dim, maxdeg, return_arr=True)
    arr_nums=arr_nums[dim, :]

    inds=np.nonzero(arr_nums==nterms)
    if len(inds[0])!=1:
        known='degdim'.replace(unknown, '')
        s=f'cannot determine `{unknown}` from `nterms` and `{known}`'
        raise ValueError(s)

    return inds[0][0]

def _num_terms_of_poly_nd(dim, deg, return_arr=False):
    '''
        number of terms in poly with `deg` and `dim`

        Let N(g, m) for num of terms of poly with `deg=g`, `dim=m`
            g>=0, m>=0
        recursion formula:
            N(g, 0) = 1
            N(g, m) = Sum_(i=0,1,..,g) N(g-i, m-1)
                    = Sum_(i=0,1,..,g) N(i, m-1)

        For order esimating, N(g, m) = O(g^m)

        N(g, m) is symmetric: N(g, m)=N(m, g)
            since N(g, m) = N(g-1, m) + N(g, m-1) for g, m >= 1
                and N(g, 0) = N(0, m) = 1

        if `retur_arr`, return array A
            for A[m, g] for poly with `deg=g`, `dim=m`
    '''
    numts=np.zeros((dim+1,deg+1), dtype=int)

    numts[0]=1
    for i in range(1, dim+1):
        numts[i]=np.cumsum(numts[i-1])

    if return_arr:
        return numts

    return numts[-1, -1]

### calculate poly terms in lexicographic order
def _poly_terms_ndim(xs, deg):
    '''
        poly terms with degree <= `deg`
            in lexicographic order

        :param xs: ndarray with shape (N, K)
            K is dimension of poly
    '''
    # arg check
    assert deg>=0, f'poly cannot have negative degree: {deg}'

    assert isinstance(xs, np.ndarray), f'unexpected `xs` type: {type(xs)}'
    assert xs.size>0, 'no `xs` to compute poly terms'
    assert xs.ndim==2, f'got unexpected `xs` array dim: {xs.ndim}'

    # powers of `xs`
    xs_pows=[np.ones_like(xs)]
    for _ in range(deg):
        xs_pows.insert(0, xs_pows[0]*xs)

    return _poly_terms_buf(xs_pows)

def _poly_terms_buf(xs_pows):
    '''
        real work to caculate poly itms in lexicographic order
            for given list of powers of `xs`

        :param xs_pows: list of 2d array with shape (N, K)
            with order for decreasing power exponent
    '''
    if xs_pows[0].shape[-1]==1:  # dim==1
        return [x[:, 0] for x in xs_pows]

    terms=[]
    deg=len(xs_pows)-1
    for i in range(deg+1):
        x0=xs_pows[i][:, 0]
        for xi in _poly_terms_buf([x[:, 1:] for x in xs_pows[-(i+1):]]):
            terms.append(x0*xi)

    return terms

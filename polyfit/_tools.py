#!/usr/bin/env python3

'''
    tools for polynomial
'''

import numbers

import numpy as np

# multidegrees: power exponents of terms
def get_multidegs(dim, deg, order=None, descending=True):
    '''
        get multidegrees in multi-dimension poly
            that is (a1, a2, ..., an) for a term x1^a1 x2^a2 ... xn^an

        return list of multidegree tuple in lexicographic order

        :param order: None, str, or callable
            if None, use lexicographic order by default

            str: 'lex', 'deglex'
                lex: lexicographic
                deglex: degree-lexicographic

            callable: used as arg `key` in function `sorted`
    '''
    if dim<=0:
        return [()]
    # if deg<=0:
    #     return [tuple([0]*dim)]

    degs=[]
    for i in range(0, deg+1):
        a1=deg-i
        for ai in get_multidegs(deg=i, dim=dim-1):
            degs.append((a1, *ai))

    # order
    if order is not None:
        return sort_degs(degs, order=order, descending=descending)

    if not descending:
        degs=degs[::-1]
    return degs

## sort iterms by power exponents
def sort_list(seq, argsort=False, key=None, **kwargs):
    '''
        support argsort
    '''
    if argsort:
        seq=list(enumerate(seq))
        if key is not None:
            key=lambda t, f0=key: f0(t[1])
        else:
            key=lambda t: t[1]

    seq=sorted(seq, key=key, **kwargs)

    if argsort:
        seq=[t[0] for t in seq]

    return seq

def sort_degs_lex(degs, descending=True, argsort=False):
    '''
        sort degs in lexicographic order
    '''
    return sort_list(degs, reverse=descending, argsort=argsort)

def sort_degs_deglex(degs, descending=True, argsort=False):
    '''
        sort degs in degree-lexicographic order
    '''
    return sort_list(degs, reverse=descending, argsort=argsort,
                           key=lambda t: (sum(t), t))

def sort_degs(degs, order='lex', descending=True, **kwargs):
    '''
        sort multi degrees

        :param order: None, str, or callable
            if None, use lexicographic order by default

            str: 'lex', 'deglex'
                lex: lexicographic
                deglex: degree-lexicographic

            callable: used as arg `key` in function `sorted`
    '''
    if isinstance(order, str):
        assert order in ['lex', 'deglex']
        if order=='deglex':
            return sort_degs_deglex(degs, descending=descending, **kwargs)
        return sort_degs_lex(degs, descending=descending, **kwargs)

    return sort_list(degs, key=order, reverse=descending, **kwargs)

## num of poly terms
def check_poly_ncoeffs_dim_deg(ncoeffs, dim=None, deg=None):
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
        numt=num_terms_of_poly_nd(dim, deg)
        if ncoeffs!=numt:
            raise ValueError('num of `coeffs` not match with that of poly terms')

        return dim, deg

    # one is not None:
    if dim is not None:  # deg None
        deg=find_unknown_for_nterms(ncoeffs, dim, unknown='deg')
    else:  # deg not None, dim None
        dim=find_unknown_for_nterms(ncoeffs, deg, unknown='dim')

    return dim, deg

def find_unknown_for_nterms(nterms, dim, unknown='deg'):
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
    arr_nums=num_terms_of_poly_nd(dim, maxdeg, return_arr=True)
    arr_nums=arr_nums[dim, :]

    inds=np.nonzero(arr_nums==nterms)
    if len(inds[0])!=1:
        known='degdim'.replace(unknown, '')
        s=f'cannot determine `{unknown}` from `nterms` and `{known}`'
        raise ValueError(s)

    return inds[0][0]

def num_terms_of_poly_nd(dim, deg, return_arr=False):
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

# for numpy.ndarray
def align_arrs_shape(xs):
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

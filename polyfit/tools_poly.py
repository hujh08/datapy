#!/usr/bin/env python3

'''
    tools for polynomial
'''

import numbers

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

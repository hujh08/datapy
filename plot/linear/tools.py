#!/usr/bin/env python3

'''
    helpful tools
'''

import numbers

import numpy as np

# functions for list-like objects
def insert_iter_to_list(container, index, iterable, inplace=True):
    '''
        insert an iterable to a list-like container
            before an `index`

        return a list
        if `inplace`, change would happen inplace
            and return `container` itself
    '''
    vals=[*container[:index], *iterable, *container[index:]]
    if inplace:
        assert type(container) is list, 'only allow list for inplace insert'
        container[:]=vals
        return container

    return vals

# product of iters
def iter_prod(*iters):
    '''
        iter along product of multi-iterators

        no yield when no args given
    '''
    if iters:
        n=len(iters)
        for vs in _iter_prod(iters):
            yield vs[0] if n==1 else vs

## auxiliary
def _iter_prod(iters):
    '''
        recursion to yield product of iters
    '''
    if not iters:
        yield ()
    else:
        iter0, *iters=iters
        for v0 in iter0:
            for vs in _iter_prod(iters):
                yield (v0, *vs)

# compare two sets
def is_same_set(s0, s1):
    '''
        whether two sets have same content
    '''
    return not set(s0).symmetric_difference(s1)

# comparator with precise
class PrecisceComparator:
    '''
        comparator with precise

        for 2 values, v1, v2
            if abs(v1-v2)<=precise, they should be viewed equal
    '''
    def __init__(self, precise=None):
        '''
            init

            Parameters:
                precise: float, or None
                    if None, use precise 0,
                        that is normal precise
        '''
        if precise is None:
            precise=0

        self._precise=precise

    # compare with zero
    def is_zero(self, v):
        '''
            whether it is zero within precise

            support number and ndarray for input `v`
            if ndarray, return a boolean array
        '''
        return np.abs(v)<=self._precise

    def is_nonzero(self, v):
        return np.logical_not(self.is_zero(v))

    def all_zeros(self, a):
        '''
            whether elements in array are all zeros
        '''
        return np.all(self.is_zero(a))

    def inds_nonzero(self, a):
        '''
            return indices of nonzero elements

            if array is 1d, return an index array
            otherwise, return tuple with len same as array's dimension
        '''
        a=np.asarray(a)

        inds=np.nonzero(self.is_nonzero(a))

        if a.ndim==1:
            return inds[0]
        return inds

    ## inequality
    def is_le_zero(self, v):
        '''
            wheter less than or equal to 0 within precise
                `v` <= 0
        '''
        return v<=self._precise

    def is_ge_zero(self, v):
        '''
            wheter greater than or equal to 0 within precise
                `v` >= 0
        '''
        return -v<=self._precise

    # filter
    def filter(self, a):
        '''
            filter iterable or scalar
            
            set vals within precise zero
        '''
        if isinstance(a, numbers.Number):
            return 0 if self.is_zero(a) else a

        return [0 if self.is_zero(k) else k for k in a]

## normal comparator
NormComp=PrecisceComparator()

# indent str
def get_indent(indent=None):
    '''
        return str for indent

        support constructing indent by giving int
    '''
    if indent is None:
        return ''
        
    if isinstance(indent, numbers.Integral):
        indent=' '*indent
    elif not isinstance(indent, str):
        raise TypeError('only allow int or str as `indent`')
    return indent
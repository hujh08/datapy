#!/usr/bin/env python3

'''
    class for multi-dimension poly
        that is poly in k[x1, x2, ..., xn]
'''

import numbers

import numpy as np

from ._tools import (get_multidegs, sort_degs)

class NDPoly:
    '''
        simple class for muti-dimensional poly

        complicated function could be got through `sympy`
            converter to `sympy` object exists
    '''
    def __init__(self, degs, coeffs=1, dim=None, **kwargs):
        '''
            init of multi-dimensinal poly

            Baisically, it is to specify items in poly
                with form c x1^a1 x2^a2 ... xn^an
                    where c: coefficient
                          (a1, ..., an): multi-degrees
            poly degree: max (Sum_i ai) along all items

            Parameters:
                degs: int, or list of multi-degrees
                    specify multi-degrees in poly

                    if number, it is degree of poly
                        then all multi-degrees below that would be used
                            with some order given by `order` in kwargs
                                lexicographic by default

                    if list of multi-degrees,
                        it could be list of int or list of tuple
                            - list of int or tuple:
                                0 deg in end could be missed
                                int d for tuple (d,)

                coeffs: number or list of arrays
                    list of coefficients
                    same length as "real" `degs`

                    if number, use same coeff for all items

                    if list of arrays,
                        len equal to number of terms

                        support ndarray coeff
                            for multiply polys

                dim: None or int
                kwargs: optional args for `get_multidegs`
                    they work only when `degs` is int
                        used to specify multi-degrees
                            for poly with a degree and a dimension

                    if None, use default:
                        dim: 1
        '''
        # multi-degrees
        if isinstance(degs, numbers.Number):
            if dim is None:
                dim=1
            degs=get_multidegs(dim, degs, **kwargs)
        else:
            degs=list(degs)
            if len(degs)==0:
                raise ValueError('at least 1 item in poly')

            degs1=[]
            for d in degs:
                if isinstance(d, numbers.Number):
                    degs1.append((d,))
                    continue

                if not all([isinstance(t, numbers.Number) for t in d]):
                    raise ValueError(f'got invalid multi-degree: {d}')

                degs1.append(tuple(d))

            dim=max(map(len, degs1))
            degs=[]
            for d in degs1:
                if len(d)<dim:
                    d+=(0,)*(dim-len(d))
                degs.append(d)

        # coeffs
        if isinstance(coeffs, numbers.Number):
            coeffs=[coeffs]*len(degs)
        else:
            coeffs=list(coeffs)
            if len(coeffs)!=len(degs):
                raise ValueError('len of `coeffs` mismatch with that of `degs`')

        # set properties
        self._degs=degs
        self._dim=dim

        self._coeffs=coeffs

        ## property facilitating calculation
        self._maxdegs=[max(d) for d in zip(*degs)]  # max deg for each variable

    def copy(self):
        return type(self)(self._degs, coeffs=self._coeffs)

    @property
    def coeffs(self):
        return list(self._coeffs)

    # reduce terms
    def reduce_like_terms(self):
        '''
            combine like terms, which have same deg
        '''
        degs_uniq=[]
        inds={}  # index in `degs_uniq`
        uniq=True
        for d in self._degs:
            if d not in inds:
                inds[d]=len(degs_uniq)
                degs_uniq.append(d)
            else:
                uniq=False

        if uniq:
            return

        coeffs=[0]*len(degs_uniq)
        for d, c in zip(self._degs, self._coeffs):
            i=inds[d]
            coeffs[i]+=c

        self._degs=degs_uniq
        self._coeffs=coeffs

    def remove_zero_terms(self):
        '''
            remove terms with zero coeffs
        '''
        if len(self._degs)==1: # only one term
            return

        if all([c!=0 for c in self._coeffs]):
            return

        degs=[]
        coeffs=[]
        for d, c in zip(self._degs, self._coeffs):
            if c==0:
                continue

            degs.append(d)
            coeffs.append(c)

        # all zero
        if not degs:
            degs.append((0,)*self._dim)
            coeffs.append(0)

        self._degs=degs
        self._coeffs=coeffs

        self._maxdegs=[max(d) for d in zip(*degs)]  # max deg for each variable

    # sort items by multi-degrees
    def sort_items(self, order='lex', descending=True):
        '''
            sort items by multi-degrees

            :param order: None, str, or callable
                if None, use lexicographic order by default

                str: 'lex', 'deglex'
                    lex: lexicographic
                    deglex: degree-lexicographic

                callable: used as arg `key` in function `sorted`
        '''
        inds=sort_degs(self._degs, order=order,
                        descending=descending, argsort=True)

        self._degs=[self._degs[i] for i in inds]
        self._coeffs=[self._coeffs[i] for i in inds]

    # evaluate
    def calc_xterms(self, *xs, broadcast=False):
        '''
            calculate terms x1^a1 * x2^a2 * ... * xn^an
                for (a1, a2, ..., an) in degs
        '''
        if len(xs)!=self._dim:
            raise ValueError('num of args mismatch with poly dim')

        if self._dim==0:
            return [1]*len(self._degs)

        # powers of xs
        pows=[]
        for maxd, xi in zip(self._maxdegs, xs):
            xpows=[1]
            for _ in range(maxd):
                xpows.append(xi*xpows[-1])
            pows.append(xpows)

        # xterms
        fpow=lambda xj, i: pows[xj][i]
        inds=list(range(self._dim))
        terms=self._gen_calc_xterms(*inds, fpow=fpow)

        if broadcast:
            terms=np.broadcast_arrays(*terms)

        return terms

    def eval(self, *xs, coeffs=None):
        '''
            evaluate poly by assign variables to some value
        '''
        if len(xs)!=self._dim:
            raise ValueError('num of args mismatch with poly dim')

        if coeffs is None:
            coeffs=self._coeffs
        elif len(coeffs)!=len(self._degs):
            s='mismatch len between `coeffs` and degs'
            raise ValueError(s)

        xterms=self.calc_xterms(*xs)
        terms=[c*t for c, t in zip(coeffs, xterms)]

        return sum(terms)

    def __call__(self, *xs, coeffs=None):
        return self.eval(*xs, coeffs=coeffs)

    ## auxiliary functions
    def _gen_calc_xterms(self, *xs, fpow=None, fprodsum=None, fprod=None):
        '''
            general calculation of xterms
                by specified function to do power, sum by product
        '''
        if len(xs)!=self._dim:
            raise ValueError('num of args mismatch with poly dim')

        if fpow is None:
            fpow=lambda x, i: x**i
        if fprodsum is None:
            import functools
            if fprod is None:
                fprod=lambda x, y: x*y
            fprodsum=lambda xs: functools.reduce(fprod, xs)

        # terms
        xterms=[]
        for tdeg in self._degs:
            xpows=[]
            for xi, di in zip(xs, tdeg):
                xpows.append(fpow(xi, di))

            xterms.append(fprodsum(xpows))

        return xterms

    # fit
    def fit(self, ys, *xs, rcond=None, inplace=True):
        '''
            adopt coeffs by fit to (*xs, ys)
        '''
        xterms=self.calc_xterms(*xs, broadcast=True)
        xterms=np.column_stack(xterms)

        # fit
        fitres=np.linalg.lstsq(xterms, ys, rcond=rcond)
        coeffs=fitres[0]

        if inplace:
            self._coeffs[:]=coeffs

        return fitres

    # partial derivative
    def deriv_1st_at(self, ind, coeffs=None):
        '''
            1st derivative at nth variate

            return new instance for it
        '''
        if coeffs is None:
            coeffs=self._coeffs
        elif len(coeffs)!=len(self._degs):
            s='mismatch len between `coeffs` and degs'
            raise ValueError(s)

        degs_d=[]
        coeffs_d=[]
        for tdeg, c in zip(self._degs, coeffs):
            if tdeg[ind]==0:
                continue

            coeffs_d.append(tdeg[ind]*c)

            tdeg=list(tdeg)
            tdeg[ind]-=1
            degs_d.append(tuple(tdeg))

        return type(self)(degs=degs_d, dim=self._dim,
                            coeffs=coeffs_d)

    # complicated functions by `sympy`
    def to_sym_poly(self):
        '''
            to `sympy.poly` instance
        '''
        import sympy

        # symbols for variables x1, ..., xn
        symbs=sympy.symbols(self._get_symbs())

        return self.eval(*symbs)

    def _get_symbs(self):
        '''
            return list of symbols for variables
        '''
        n=self._dim
        if n>26:
            symbs=[f'x{i}' for i in range(n)]
        else:
            s0='x' if n<=3 else 'a'
            symbs=[chr(ord(s0)+i) for i in range(n)]

        return symbs

    # stringlization
    def __repr__(self):
        sname=type(self).__name__
        sdegs=repr(self._degs)
        scoeffs=repr(self._coeffs)
        return f'{sname}({sdegs}, coeffs={scoeffs})'

    def __str__(self):
        symbs=self._get_symbs()

        # functions
        fprodsum=lambda ts: '*'.join([t for t in ts if t])
        fpow=lambda x, i: '' if i==0 else (x if i==1 else f'{x}^{i}')
        xterms=self._gen_calc_xterms(*symbs, fpow=fpow, fprodsum=fprodsum)

        terms=[]
        for c, xt in zip(self._coeffs, xterms):
            if c==0:
                continue

            t=fprodsum(['' if c==1 else str(c), xt])
            terms.append(t if t else '1')

        if not terms:
            return '0'

        return ' + '.join(terms)

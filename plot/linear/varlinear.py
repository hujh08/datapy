#!/usr/bin/env python3

'''
    module to handle variables and linear constraints among them

    Some variables are selected as basis, like (v1, v2, .., vn)
    Others are linear combination of them (with a bias):
        v = k1*v1 + ... + kn*vn + c
'''

import numbers

import numpy as np

from .bases import LnComb, LnCoeffs, LnConflictError
from .lneq import LnEqs
from .tools import insert_iter_to_list, is_same_set, PrecisceComparator

class LinearManager:
    '''
        manager of linear constraints of variables
    '''
    def __init__(self, size=int(1e3), precise=1e-10):
        '''
            initiation of manager

            Parameters:
                size: int, None
                    limit on max num of variables in control
                    Too many variables would exhaust memory and tug calculation

                    If None, no limit
                        cautious to set

                precise: non-negative float, or None
                    number less than (or equal to) this number is thought as zero
                    It is also used to compare two number via abs of difference
                        k0==k1 if abs(k0-k1)<=precise

                    if None, use normal equality
                        same as `precise`=0
        '''
        # comparator
        if not isinstance(precise, PrecisceComparator):
            precise=PrecisceComparator(precise)
        self._comp=precise   # comparator

        # equality constraints
        self._eqs=LnEqs(size, self._comp)

    # copy
    def copy(self):
        obj=self.__class__()
        obj._comp=self._comp
        obj._eqs=self._eqs.copy()

        return obj

    # add linear constraint
    def add_vars_lcons(self, vs, coeffs=1, const=0):
        '''
            add linear constraint of variables
                k1*v1+k2*v2+...+kn*vn+c=0
            where v1, .., vn are variables to constrain

            Parameters:
                vs: iterable
                    variables to constrain

                coeffs: number or iterable
                    coeffs, [k1, .., kn]

                    if number, use `len(vs)` to extend as list

                const: number
                    const
        '''
        # reduce expression
        vstot, kstot, ctot=self._eqs.reduce_express_of(vs, coeffs, const)

        # nonz
        inds_nonz=list(self._comp.inds_nonzero(kstot))

        # identical eq: check conflict
        if len(inds_nonz)==0:
            ftest=self._comp.is_zero
            if not ftest(ctot):
                raise LnConflictError('conflict in adding cons')
            return

        # vars with nonzero coeff exist
        ilz=inds_nonz.pop()  # index of last nonz
        vlz=vstot.pop(ilz)
        klz=kstot.pop(ilz)

        ks=np.array(kstot)
        nb0=self._eqs.get_len_basis()

        ## add other unknown vars as basis, if exists
        self._eqs.add_new_basis_vars(vstot[nb0:])

        # constant
        if not inds_nonz:
            # new constant
            if ilz>=nb0:  # unknown var, skip new basis adding
                self._eqs.add_new_const_var(vlz, -ctot/klz)
            else:         # basis to const
                self._eqs.set_basis_var_const(ilz, -ctot/klz,
                                               use_ind=True)

            return

        # unknown vars exist
        if ilz>=nb0:
            self._eqs.add_new_lcomb_var(vlz, -ks/klz, -ctot/klz)
            return

        # no unknown vars
        self._eqs.add_eq_to_basis_var(ilz, -ks/klz, -ctot/klz,
                                       use_ind=True)

    ## use LnComb, more user-friendly
    def add_lncomb(self, left, right=0):
        '''
            add constraint written as LnComb object
                left = right
                where both `left` and `right` are LnComb-like object
        '''
        left=LnComb.lncomb(left)
        right=LnComb.lncomb(right)

        p=left-right

        vs, ks, c=p.get_vs_ks_c()
        self.add_vars_lcons(vs, ks, c)

    # evaluate based on the linear constraint
    def reduce_express_of_lncomb(self, lncomb):
        '''
            return reduced expression of lncomb

            see `LnEqs.reduce_express_of` for detail
        '''
        lncomb=LnComb.lncomb(lncomb)
        vs, ks, c=lncomb.get_vs_ks_c()

        return self._eqs.reduce_express_of(vs, ks, c)

    def eval_lncomb(self, lncomb):
        '''
            evaluate a linear combination

            if not a constant result, return None
        '''
        # reduce based on eqs
        vs, ks, c=self.reduce_express_of_lncomb(lncomb)

        if not self._comp.all_zeros(ks):
            return None

        return c

    def eval_ratio_of_lncombs(self, t0, t1):
        '''
            return ratio of linear combinations, t0/t1

            if not constant ratio, return None
        '''
        # reduce
        vs0, ks0, c0=self.reduce_express_of_lncomb(t0)
        vs1, ks1, c1=self.reduce_express_of_lncomb(t1)

        # exists different unknown vars with nonzero coeff
        if len(vs0)!=len(vs1):
            return None

        # same length
        nb=self._eqs.get_len_basis()
        if len(vs0)>nb:
            # different vs set
            if not is_same_set(vs0[nb:], vs1[nb:]):
                return None

            # to same order of vars
            map_vk1=dict(zip(vs1[nb:], ks1[nb:]))
            ks1=ks1[:nb]
            for v in vs0[nb:]:
                ks1.append(map_vk1[v])

        # compute ratio
        ksc0, ksc1=map(np.asarray, [[*ks0, c0], [*ks1, c1]])

        inzs0=self._comp.inds_nonzero(ksc0)
        inzs1=self._comp.inds_nonzero(ksc1)
        inzs=[*inzs0, *inzs1]

        if not inzs:  # all zeros
            ratios=(c0, c1)  # ratio: c0/c1
        else:
            i=inzs[-1]
            k0, k1=ksc0[i], ksc1[i]

            # not ratio
            if not self._comp.all_zeros(k1*ksc0-k0*ksc1):
                return None

            ratios=(k0, k1)

        return ratios[0]/ratios[1]

    # print
    def info(self, indent=' '*4):
        '''
            print equations in manager
        '''
        # indent
        if isinstance(indent, numbers.Integral):
            indent=' '*indent
        elif not isinstance(indent, str):
            raise TypeError('only allow int or str as `indent`')

        # print
        vbs=self._eqs.get_list_basis()

        print('eqs:')
        for v, (ks, c) in self._eqs.iter_lcomb_items():
            *ks, c=self._comp.filter([*ks, c])
            print(indent+'%s = %s' % (v, LnComb(vbs, ks, c)))

        for v, c in self._eqs.iter_const_items():
            c=self._comp.filter(c)
            print(indent+'%s = %s' % (v, LnComb(const=c)))

        print('basis:', ', '.join(vbs))

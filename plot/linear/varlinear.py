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
from .lnineq import LnIneqs
from .tools import insert_iter_to_list, is_same_set, get_indent, \
                   PrecisceComparator

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

        # inequality constraints
        self._ineqs=LnIneqs(self._comp)

    # copy
    def copy(self):
        obj=self.__class__()
        obj._comp=self._comp
        obj._eqs=self._eqs.copy()
        obj._ineqs=self._ineqs.copy()

        return obj

    # handle vars
    def _append_new_basis_vars(self, vs):
        '''
            append new basis vars in end
        '''
        self._eqs.append_new_basis_vars(vs)

        # ineqs
        self._ineqs.append_new_vars(len(vs))

    def _append_new_basis_var(self, v):
        self._append_new_basis_vars([v])

    def _add_new_lcomb_var(self, v, ks, c):
        '''
            add new lcomb var
        '''
        self._eqs.add_new_lcomb_var(v, ks, c)

    def _add_new_const_var(self, v, c):
        '''
            add new const var
        '''
        self._eqs.add_new_const_var(v, c)

    ## constraint to basis
    ### inequality constraint
    def _add_ineq_bound(self, ksc, upper=True):
        '''
            add ineq bound
        '''
        self._ineqs.add_new_bound_coop_with(self._eqs, ksc, upper)

    ### equality constraint
    def _add_eq_to_basis_var(self, ksc):
        '''
            set eq constraint to basis var
        '''
        self._ineqs.add_degen_coop_with(self._eqs, ksc)

    # add linear constraint
    def add_vars_lcons(self, vs, coeffs=1, const=0, ineq=False):
        '''
            add linear constraint of variables
                k1*v1+k2*v2+...+kn*vn+c=0 or <=0
            where v1, .., vn are variables to constrain

            Parameters:
                vs: iterable
                    variables to constrain

                coeffs: number or iterable
                    coeffs, [k1, .., kn]

                    if number, use `len(vs)` to extend as list

                const: number
                    const

                ineq: bool, default: False
                    whether inequality

                    if True, add ineq
                        k1*v1+k2*v2+...+kn*vn+c <= 0
        '''
        # reduce expression
        vstot, kstot, ctot=self._eqs.reduce_express_of(vs, coeffs, const)

        # nonz
        inds_nonz=list(self._comp.inds_nonzero(kstot))

        # identical eq: check conflict
        if len(inds_nonz)==0:
            if ineq:  # 0<=c
                ftest=self._comp.is_le_zero
            else:
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

        ksc=-LnCoeffs(ks, ctot)/klz

        ## add other unknown vars as basis, if exists
        self._append_new_basis_vars(vstot[nb0:])

        # inequality
        if ineq:
            if ilz>=nb0:
                self._append_new_basis_var(vlz)

            upper=self._comp.is_ge_zero(klz)
            self._add_ineq_bound(ksc[:ilz], upper)

            return

        # unknown vars exists
        if ilz>=nb0:
            if not inds_nonz:  # constant
                self._add_new_const_var(vlz, ksc.const)
            else:
                self._add_new_lcomb_var(vlz, *ksc)

            return

        # no unknown vars, constraints between basis
        self._add_eq_to_basis_var(ksc[:ilz])

    ## use LnComb, more user-friendly
    def add_lncomb(self, left, right=0, ineq=False):
        '''
            add constraint written as LnComb object
                left = right or 
                left <= right
                where both `left` and `right` are LnComb-like object

            :param ineq: bool, or str, default: False
                whether inequality

                if True, add upper ineq: left <= right

                for str, only support:
                    upper: 'upper', 'le', '<='
                    lower: 'lower', 'ge', '>='

        '''
        # str for ineq
        if type(ineq) is str:
            sge=['upper', 'le', '<=']
            sle=['lower', 'ge', '>=']
            if ineq in sge:
                upper=True
            elif ineq in sle:
                upper=False
            else:
                raise ValueError('unexpected `ineq`, '
                    'only allow %s and %s' % (repr(sge), repr(sle)))
            ineq=True
        elif ineq:
            upper=True  # use upper as default

        # lncomb
        left=LnComb.lncomb(left)
        right=LnComb.lncomb(right)

        p=left-right
        if ineq and not upper:
            p=-p

        vs, ks, c=p.get_vs_ks_c()
        self.add_vars_lcons(vs, ks, c, ineq=ineq)

    def add_lncomb_ineq(self, left, right=0, upper=True):
        '''
            add ineq:
                left <= right or
                left >= right

            :param upper: bool
                if True, add: left <= right
        '''
        if not upper:
            left, right=right, left
        self.add_lncomb(left, right, ineq=True)

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

    ## eval bounds
    def eval_bounds_of_lncomb(self, lncomb):
        '''
            eval lower and upper bounds of a LnComb
        '''
        # reduce based on eqs
        _, ks, c=self.reduce_express_of_lncomb(lncomb)

        indsnz=self._comp.inds_nonzero(ks)

        # const
        if len(indsnz)==0:
            return c, c

        # exists unknown vars
        nb=self._eqs.get_len_basis()
        if len(ks)>nb:
            return None, None

        # remove zeros from tail
        ilz=indsnz[-1]
        ksc=LnCoeffs(ks[:(ilz+1)], c)
        return self._ineqs.eval_bounds_of(ksc)

    # print
    def info(self, indent_glob=None, indent=' '*4):
        '''
            print equations in manager
        '''
        # indent
        gindent=get_indent(indent_glob)  # global indent
        indent=gindent+get_indent(indent)

        # print
        vbs=self._eqs.get_list_basis()

        ## eqs
        print(gindent+'eqs:')
        print(self._eqs.get_str_of_eqs(indent))

        ## ineqs
        print(gindent+'ineqs:')
        print(self._ineqs.get_str_of_ineqs(indent=indent, vs=vbs))

        ## basis
        print(gindent+'basis:', ', '.join(vbs))

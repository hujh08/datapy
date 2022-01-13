#!/usr/bin/env python3

'''
    base class for linear equality constraints
'''

import numbers

from .bases import LnCoeffs
from .tools import PrecisceComparator, insert_iter_to_list

class LnEqs:
    '''
        base class for linear equalities
    '''
    def __init__(self, size=int(1e3), precise=None):
        '''
            init

            Parameters:
                size: int, None
                    limit on max num of variables in control
                    Too many variables would exhaust memory and tug calculation

                    If None, no limit
                        be careful to set it

                precise: PrecisceComparator, float, or None
                    used to create comparator with precise
        '''
        # comparator
        if not isinstance(precise, PrecisceComparator):
            precise=PrecisceComparator(precise)
        self._comp=precise   # comparator

        # size
        if size is not None:
            if not isinstance(size, numbers.Integral):
                raise TypeError('only allow int as `size`')

            if size<2:
                raise ValueError('too small size, at least 2')

        self._size=size

        # container of vars
        self._init_vars_container()

    def _init_vars_container(self):
        '''
            init container of variables

            3 types:
                - basis vars:
                    (v1, v2, ..., vn)

                - lcomb vars: linear combination of basis (with a bias):
                    k1*v1 + k2*v2 + ... + kn*vn + c

                - const vars: constant
                    special lcomb, just with all zeros coeffs
        '''
        # basis vars
        self._basis_vars=[]

        # linear combination:
        self._lcomb_vars={}

        # constant vars: {d: c}
        self._const_vars={}

    # copy
    def copy_vars_from(self, other):
        '''
            copy vars from another object

            ignore comparator
        '''
        if not isinstance(other, self.__class__):
            raise TypeError('cannot copy from other type')

        # # comparator
        # self._comp=other._comp

        # size check
        if self._size is not None:
            n1=other.get_len_vars()
            self._confirm_not_off_size(n1)

        # basis vars
        self._basis_vars[:]=other._basis_vars

        # lcomb
        self._lcomb_vars.clear()
        for v, ksc in other._lcomb_vars.items():
            self._lcomb_vars[v]=ksc.copy()

        # const
        self._const_vars.clear()
        for v, c in other._const_vars.items():
            self._const_vars[v]=c

    def copy(self):
        obj=self.__class__(self._size, self._comp)
        obj.copy_vars_from(self)

        return obj

    # size check
    def _confirm_not_off_size_for_new_vars(self, n=1):
        '''
            confirm not off size when add `n` new vars
        '''
        if self._size is None or n<=0:
            return

        ntot=self.get_len_vars()
        self._confirm_not_off_size(ntot+n)

    def _confirm_not_off_size(self, s):
        '''
            confirm size `s` not off size
        '''
        if self._size is None:
            return

        if s>self._size:
            raise MemoryError('too many variables, '
                              'at most %i' % self._size)

    ## size of total vars
    def get_len_vars(self):
        '''
            return total number of vars
        '''
        return len(self._basis_vars) + \
               len(self._lcomb_vars) + \
               len(self._const_vars)

    # variable type
    def get_var_type(self, v):
        '''
            return type of variable
                basis, lcomb, const, or 
                None, for not contained
        '''
        if v in self._basis_vars:
            return 'basis'  # basis

        if v in self._lcomb_vars:
            return 'lcomb'  # linear combination

        if v in self._const_vars:
            return 'const'  # constant

        return None  # not contained

    def contains(self, v):
        # contains a variable
        return self.get_var_type(v) is not None

    def is_basis_var(self, v):
        return self.get_var_type(v)=='basis'
    def is_lcomb_var(self, v):
        return self.get_var_type(v)=='lcomb'
    def is_const_var(self, v):
        return self.get_var_type(v)=='const'

    # lcomb and const var
    def add_new_lcomb_var(self, v, coeffs, const):
        '''
            add new lcomb var

            if all zeros in `coeffs`, add as const var

            ==========
            Parameters:
                v: variable
                    variable to add

                coeffs: number or iterable
                    coeffs, ks=[k1, ..., kn]

                    if number, use len of basis to extend to list

                const: number
                    const bias, c
        '''
        self._confirm_not_off_size_for_new_vars()

        assert not self.contains(v), \
            'var already exists: %s' % v

        # ksc
        nb=len(self._basis_vars)
        if isinstance(coeffs, numbers.Number):
            coeffs=[coeffs]*nb
        ksc=LnCoeffs(coeffs, const)  # type check done in `LnCoeffs`
        ks, c=ksc

        if len(ks)!=nb:
            raise ValueError('mismatch between len of coeffs and basis')

        if self._comp.all_zeros(ks):
            self._const_vars[v]=c
            return

        self._lcomb_vars[v]=ksc

    def add_new_const_var(self, v, c):
        '''
            add new const var
        '''
        self._confirm_not_off_size_for_new_vars()

        assert not self.contains(v), \
            'var already exists: %s' % v

        if not isinstance(c, numbers.Number):
            raise TypeError('invalid type for const, only allow number')

        self._const_vars[v]=c

    # basis var
    def add_new_basis_vars(self, vs, index='tail'):
        '''
            add new basis vars before an index
            return index and size inserted

            Parameters:
                vs: iterable
                    list of variables to add as basis

                index: int, 'head' or 'tail'
                    position of new vars
        '''
        # iterable vs
        vs=list(vs)
        num=len(vs)

        if num==0:
            return

        self._confirm_not_off_size_for_new_vars(num)

        ## must not exists
        if any([self.contains(v) for v in vs]):
            raise ValueError('some vars in `vs` already exists')

        # index
        if index=='tail':
            index=len(self._basis_vars)
        elif index=='head':
            index=0
        elif not isinstance(index, numbers.Integral):
            raise TypeError(
                'only allow int, \'head\', \'tail\' as `index`')

        # insert new vars
        insert_iter_to_list(self._basis_vars, index, vs, inplace=True)

        # modify lcomb
        for ksci in self._lcomb_vars.values():
            ksci.insert_zeros(index, num)

        return index, num

    def add_new_basis_var(self, v, index='tail'):
        '''
            add one basis var
        '''
        return self.add_new_basis_vars([v], index)

    def add_eq_to_basis_var(self, v, coeffs, const, use_ind=True):
        '''
            add eq constraint to basis var
                vi = k1*v1 + .. +
                     k(i-1)*v(i-1) + k(i+1)*v(i+1) + .. +
                     kn*vn + c
            this var would become lcomb var
                expressed by left basis vars

            ===========
            Parameters:
                v: variable or int
                    variable name or index for basis var to change

                coeffs: number or iterable
                    coeffs, ks=[k1, ..., kn]

                    if number, use len of new basis to extend to list

                const: number
                    const bias, c

                use_ind: bool
                    whether the given `v` is index or variable name
                    True for index

            Return:
                index of var, and
                LnCoffs(ks, c)
        '''
        if not use_ind:
            ind=self._basis_vars.index(v)
        else:
            ind=v

        # pop
        v=self._basis_vars.pop(ind)

        # ksc
        nb1=len(self._basis_vars)    # len of new basis
        if isinstance(coeffs, numbers.Number):
            coeffs=[coeffs]*nb1
        ksc=LnCoeffs(coeffs, const)
        if len(ksc)!=nb1:
            raise ValueError('mismatch between len of coeffs and new basis')

        # modify lcomb vars
        lcombs=list(self._lcomb_vars.keys())
        for vi in lcombs:
            ksci=self._lcomb_vars[vi]

            k0=ksci.pop(ind)
            ksci+=ksc*k0

            # zeros coeffs, change to const var
            if self._comp.all_zeros(ksci.coeffs):
                self._lcomb_vars.pop(vi)
                self._const_vars[vi]=ksci.const

        # add as new lcomb var
        if self._comp.all_zeros(ksc.coeffs):
            self._const_vars[v]=ksc.const
        else:
            self._lcomb_vars[v]=ksc

        return ind, ksc

    def set_basis_var_const(self, v, const, use_ind=True):
        '''
            set a basis var to const
                special linear constraint, with coeffs=0
        '''
        self.add_eq_to_basis_var(v, 0, const, use_ind=use_ind)

    # express linear combinations
    def reduce_express_of(self, vs, coeffs, const):
        '''
            Reduce a linear combination
                k1*v1+k2*v2+..+km*vm+c
            to form:
                Sum kb*vb + Sum ku*vu + c
            where (vb) is all basis vars
                  (vu) is unknown vars with nonzero coeff
            
            Note:
                unknown vars with zero coeff would drop,
                    but all basis vars are kept, even zero coeff

            ==========
            Parameters:
                vs: iterable
                    list of variables

                    all kind of vars,
                        basis, lcomb, const or unknown

                coeffs: number or iterable
                    coeffs, ks=[k1, ..., kn]

                    if number, use len of `vs` to extend to list

                const: number
                    const bias, c

            Return:
                (vs, ks, c)
        '''
        # iterable vs
        vs=list(vs)
        num=len(vs)

        # coeffs
        if isinstance(coeffs, numbers.Number):
            coeffs=[coeffs]*num
        else:
            if len(coeffs)!=num:
                raise ValueError(
                        'mismatch between len of `coeffs` and `vs`')

        # to standard form
        vstot=list(self._basis_vars)  # total vars
        nv0=len(vstot)
        kstot=[0]*nv0
        ctot=const

        for vi, ki in zip(vs, coeffs):
            if self._comp.is_zero(ki):
                continue

            if vi in vstot:
                j=vstot.index(vi)
                kstot[j]+=ki
                continue

            # const var
            if vi in self._const_vars:
                ctot+=ki*self._const_vars[vi]
                continue

            # lcomb var
            if vi in self._lcomb_vars:
                ksi, ci=self._lcomb_vars[vi]

                for j in range(nv0):
                    kstot[j]+=ki*ksi[j]
                ctot+=ki*ci
                continue

            # unknown vars
            vstot.append(vi)
            kstot.append(ki)

        # remove unknown vars with zero coeff
        for i in reversed(range(nv0, len(vstot))):
            if self._comp.is_zero(kstot[i]):
                vstot.pop(i)
                kstot.pop(i)

        return vstot, kstot, ctot

    # useful properties
    def get_len_basis(self):
        '''
            return number of basis
        '''
        return len(self._basis_vars)

    def get_list_basis(self):
        '''
            return list of basis vars
        '''
        return list(self._basis_vars)

    def iter_lcomb_items(self):
        '''
            iter along lcomb vars
        '''
        return iter(self._lcomb_vars.items())

    def iter_const_items(self):
        '''
            iter along const vars
        '''
        return iter(self._const_vars.items())

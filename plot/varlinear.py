#!/usr/bin/env python3

'''
    module to handle variants and linear constraints among them

    Some variants are selected as basis, like (v1, v2, .., vn)
    Others are linear combination of them (with a bias):
        v = k1*v1 + ... + kn*vn + c
'''

import numbers

import numpy as np

class LinearManager:
    '''
        manager of variants
        Linear constraints are traced and maintained dynamically.
    '''
    def __init__(self, size=1e4, precise=1e-10):
        '''
            initiation of manager

            Parameters:
                size: int, None
                    limit on max num of variants in control
                    Too many variants would exhaust memory and tug calculation

                    If None, no limit
                        be careful to set it

                precise: non-negative float, or None
                    number less than (or equal to) this number is thought as zero
                    It is also used to compare two number via abs of difference
                        k0==k1 if abs(k0-k1)<=precise

                    if None, use normal equality
                        same as `precise`=0
        '''
        assert precise is None or precise>=0, \
                'only allow None or positive for precise'
        if precise is None:
            precise=0
        self._precise=precise

        if size is not None:
            size=int(size)
        self._size=size

        # container of vars
        self._init_vars_container()

    def copy(self):
        manager=self.__class__(self._size, self._precise)

        return manager

    # zero or nonzero within precise
    def _is_zero(self, v):
        '''
            whether it is zero within precise

            support number and ndarray for input `v`
            if ndarray, return a boolean array
        '''
        return np.abs(v)<=self._precise

    def _is_nonzero(self, v):
        return np.logical_not(self._is_zero(v))

    def _all_zero(self, a):
        '''
            whether elements in array are all zero
        '''
        return np.all(self._is_zero(a))

    def _inds_nonzero(self, a):
        '''
            return indices of nonzero elements

            if array is 1d, return an index array
            otherwise, return tuple with len same as array's dimension
        '''
        a=np.asarray(a)

        inds=np.nonzero(self._is_nonzero(a))

        if a.ndim==1:
            return inds[0]
        return inds

    # init of variants container
    def _init_vars_container(self):
        '''
            init container of variants

            3 types:
                - basis vars:
                    (v1, v2, ..., vn)
                - lcomb vars: linear combination of basis (with a bias):
                    k1*v1 + k2*v2 + ... + kn*vn + c
                - const vars: constant
        '''
        # basis vars
        self._basis_vars=[]

        # linear combination:
        self._lcomb_vars={}

        # constant vars: {d: c}
        self._const_vars={}

    # vars size
    def _get_len_vars(self):
        '''
            return total number of contained vars
        '''
        return len(self._basis_vars) + \
               len(self._lcomb_vars) + \
               len(self._const_vars)

    def _check_size_vars(self, n=1):
        '''
            confirm not exceeding of max size
                when add `n` new vars
        '''
        if self._size is None or n<=0:
            return

        assert self._get_len_vars()+n<=self._size, \
                'too many variants, at most %i' % self._size

    # base methods: variant type test
    def _get_var_type(self, v):
        '''
            return type of variant
                basis, lcomb, const, or not contained

            basic method for other type tests
        '''
        if v in self._basis_vars:
            return 'basis'  # basis

        if v in self._lcomb_vars:
            return 'lcomb'  # linear combination

        if v in self._const_vars:
            return 'const'  # constant

        return None  # not contained

    def _contains(self, v):
        return self._get_var_type(v) is not None

    def _is_basis_var(self, v):
        return self._get_var_type(v)=='basis'
    def _is_lcomb_var(self, v):
        return self._get_var_type(v)=='lcomb'
    def _is_const_var(self, v):
        return self._get_var_type(v)=='const'

    # base methods: for basis variants
    def _get_len_basis(self):
        '''
            return number of basis vars
        '''
        return len(self._basis_vars)

    def _get_basis_var_index(self, v):
        '''
            return index of a basis var
        '''
        return self._basis_vars.index(v)

    def _get_basis_vars_list(self):
        '''
            return list of basis vars
        '''
        return list(self._basis_vars)

    ## foced change of basis: modify of lcomb vars must follow
    def _forced_add_new_basis_vars(self, vs, index='tail'):
        '''
            forced add of new vars as basis
            return index and size inserted

            incomplete process
            modifying of lcomb vars must follow immediately
                to maintain consistence of coeffs

            Parameters:
                vs: list of vars
                    new vars to add

                index: int or str 'head', 'tail'
                    index to insert before

                    if 'tail', append to the tail
                    if 'head', insert at head
                        same as index=0
        '''
        vs=list(vs)

        # only support to add new vars
        assert all([not self._contains(v) for v in vs]), \
                'some vars already exists'

        # vars size check
        nnew=len(vs)
        self._check_size_vars(nnew)

        # index
        if index=='tail':
            index=self._get_len_basis()
        elif index=='head':
            index=0
        assert isinstance(index, numbers.Integral), \
                'expect integral for index, got %s' % type(index).__name__

        # insert new vars
        bs0=self._basis_vars
        self._basis_vars[:]=bs0[:index]+vs+bs0[index:]

        return index, nnew

    def _forced_pop_basis_var(self, index):
        '''
            forced pop a basis var
            return var name

            incomplete process
            modifying of lcomb vars must follow immediately
                to maintain consistence of coeffs

            Parameters:
                index: int
                    index of var to pop
        '''
        return self._basis_vars.pop(index)

    # base methods for lcomb variants
    def _init_lcomb_var_container(self, v):
        '''
            init container for coeffs and const
                of a lcomb var `v`

            coeffs: float array with len same as basis
                initial values are all `nan`
            const: `nan`

            if already exists as lcomb, re-init `coeffs`
                not change `const`
        '''
        is_lcomb=self._is_lcomb_var(v)
        assert is_lcomb or not self._contains(v), \
                'already exists as other type of var'

        n=self._get_len_basis()
        coeffs=_get_coeffs_container(n)

        if not is_lcomb:
            self._check_size_vars()
            self._lcomb_vars[v]=dict(coeffs=coeffs, const=np.nan)
        else:
            self._lcomb_vars[v]['coeffs']=coeffs

    def _set_lcomb_var_coeffs(self, v, coeffs):
        '''
            set coeffs of lcomb var

            Parameters:
                v: variant name
                    must exist as lcomb var

                coeffs: number, or list-like of number
                    object that supports item assignment in ndarray
        '''
        assert not self._all_zero(coeffs), \
                'all zero in coeffs. set a const var instead'
        self._lcomb_vars[v]['coeffs'][:]=coeffs

    def _set_lcomb_var_const(self, v, const):
        '''
            set const of lcomb var

            Parameters:
                v: variant name
                    must exist as lcomb var

                const: number
                    constant bias
        '''
        self._lcomb_vars[v]['const']=float(const)

    def _get_lcomb_vars_list(self):
        '''
            return list of lcomb vars
        '''
        return list(self._lcomb_vars.keys())

    def _get_lcomb_var_coeffs(self, v):
        '''
            return coeffs of a lcomb var `v`
        '''
        return self._lcomb_vars[v]['coeffs']

    def _get_lcomb_var_const(self, v):
        '''
            return const of a lcomb var `v`
        '''
        return self._lcomb_vars[v]['const']

    def _get_lcomb_var_coeffs_const(self, v):
        '''
            return both coeffs and const of a lcomb var `v`
        '''
        return self._get_lcomb_var_coeffs(v), \
               self._get_lcomb_var_const(v)

    def _pop_lcomb_var(self, v):
        '''
            pop a lcom var
            return coeffs and constant bias
        '''
        kc=self._get_lcomb_var_coeffs_const(v)
        self._lcomb_vars.pop(v)
        return kc

    # base methods for const variants
    def _base_set_const_var(self, v, const):
        '''
            set const var: v = c
            base function

            if not exists, create new var

            Parameters:
                v: str
                    variant name

                const: number
                    constant bias, `c`
        '''
        is_const=self._is_const_var(v)
        assert is_const or not self._contains(v)

        if not is_const:  # new const var
            self._check_size_vars()

        self._const_vars[v]=float(const)

    def _get_const_var_val(self, v):
        '''
            return value of constant var
        '''
        return self._const_vars[v]

    def _pop_const_var(self, v):
        '''
            pop a constant var
            return its value
        '''
        c=self._get_const_var_val(v)
        self._const_vars.pop(v)
        return c

    ## upper level methods of nonbasis vars
    def _add_new_lcomb_var(self, v, coeffs, const):
        '''
            add a new 'lcomb' var

            if all zero in `coeffs`, add a new const var
        '''
        assert not self._contains(v)

        if self._all_zero(coeffs):
            self._base_set_const_var(v, const)
            return

        self._init_lcomb_var_container(v)
        self._set_lcomb_var_coeffs(v, coeffs)
        self._set_lcomb_var_const(v, const)

    # change of basis: complete process
    def _add_new_basis_vars(self, vs, index='tail'):
        '''
            complete process to add new basis
        '''
        # forced to change basis
        ind, num=self._forced_add_new_basis_vars(vs, index=index)
        ks0=[0]*num

        # modify lcom vars to maintain consistence
        keys=self._get_lcomb_vars_list()
        for vi in keys:
            ksi=list(self._get_lcomb_var_coeffs(vi))

            self._init_lcomb_var_container(vi)
            self._set_lcomb_var_coeffs(vi, ksi[:ind]+ks0+ksi[ind:])

    def _var_basis_to_lcomb(self, v, coeffs, const, is_ind_v=True):
        '''
            change basis var `v`
                to linear combination of other bases

            Parameters:
                v: var or int
                    var name or index for basis var to change

                coeffs, const: list-like and number
                    linear combination for `v` with respect to other basis
                    vi = k1*v1 + .. + k(i-1)*v(i-1) + k(i+1)*v(i+1) + .. 
                            + kn*vn + c

                is_ind_v: bool
                    whether the given `v` is index or var name
                    True for index
        '''
        if not is_ind_v:
            ind=self._get_basis_var_index(v)
        else:
            ind=v

        coeffs=np.asarray(coeffs)
        const=float(const)

        # forced to pop basis var
        v=self._forced_pop_basis_var(ind)

        # modify lcom vars to maintain consistence
        keys=self._get_lcomb_vars_list()
        for vi in keys:
            ksi, ci=self._get_lcomb_var_coeffs_const(vi)

            ksi=list(ksi)
            k0=ksi.pop(ind)
            ksi=np.array(ksi)

            ci+=k0*const
            ksi+=k0*coeffs

            if self._all_zero(ksi):
                self._pop_lcomb_var(vi)
                self._base_set_const_var(vi, ci)
                continue

            self._init_lcomb_var_container(vi)
            self._set_lcomb_var_coeffs(vi, ksi)
            self._set_lcomb_var_const(vi, ci)

        # add back as lcomb var: if all zero in coeffs, add const var
        self._add_new_lcomb_var(v, coeffs, const)

    def _var_basis_to_const(self, v, const, is_ind_v=True):
        '''
            change base var to const
        '''
        self._var_basis_to_lcomb(v, 0, const, is_ind_v=is_ind_v)

    # add linear constrant among variants
    def _add_vars_lcons(self, vs, coeffs=1, const=0):
        '''
            add linear constraint of variants
                k1*v1+k2*v2+...+kn*vn = c
            where
                v1, v2, .., vn: variants constrained
                k1, k2, .., kn: coefficients
                c: constant

            Parameters:
                vs: list of str
                    variants in constraint

                coeffs: list of number or scalar
                    use function `_fill_coeffs_to_len`
                        to array with len same as `vs`

                const: number
                    const
        '''
        coeffs=_fill_coeffs_to_len(coeffs, len(vs))
        const=float(const)

        # transform to: Sum kb*vb + Sum ku*vu = c
        #     where (vu) is unknown vars (not in manager)
        #           (vb) is base vars
        vs_bu=self._get_basis_vars_list()
        nbase0=len(vs_bu)
        ks_bu=[0]*nbase0

        for v, k in zip(vs, coeffs):
            if v in vs_bu:  # treat unknown vars as bases
                i=vs_bu.index(v)
                ks_bu[i]+=k
                continue

            # unknown vars
            if not self._contains(v):
                vs_bu.append(v)
                ks_bu.append(k)
                continue

            if self._is_const_var(v):
                c=self._get_const_var_val(v)
                const-=k*c
                continue

            # linear combination
            ks, c=self._get_lcomb_var_coeffs_const(v)
            for i, ki in enumerate(ks):
                ks_bu[i]+=k*ki
            const-=k*c

        ks_bu=np.array(ks_bu)
        # inds_nonz=np.nonzero(ks_bu)[0]
        inds_nonz=self._inds_nonzero(ks_bu)

        # identical eq: check conflict
        if len(inds_nonz)==0:
            # assert const==0
            assert self._is_zero(const), \
                    'conflict constraint meeted'
            return

        # constant
        if len(inds_nonz)==1:
            i=inds_nonz[-1]
            k=ks_bu[i]

            # new constant
            if i>=nbase0:  # unknown var
                self._base_set_const_var(vs_bu[i], const/k)
            else:          # basis to const
                self._var_basis_to_const(i, const/k)

            return

        # at least 2 vars in 'Sum kb*vb + Sum ku*vu = c'
        ks_b0=ks_bu[:nbase0]

        inds_u=[i for i in inds_nonz if i>=nbase0] # unknown vars

        ## unknown vars exist
        if len(inds_u)>=1:
            ci=inds_u.pop()  # vars as lcomb

            v=vs_bu[ci]
            k=ks_bu[ci]

            ks=ks_b0

            if len(inds_u)>=1:
                vs_u=[vs_bu[i] for i in inds_u]
                ks_u=[ks_bu[i] for i in inds_u]

                self._add_new_basis_vars(vs_u)
                ks=np.concatenate([ks, ks_u])

            # add new lcomb var
            self._add_new_lcomb_var(v, -ks/k, const/k)

            return

        # no unknown vars
        i=inds_nonz[-1]
        k=ks_b0[i]

        ks=np.concatenate([ks_b0[:i], ks_b0[(i+1):]])
        self._var_basis_to_lcomb(i, -ks/k, const/k)

    ## user method
    def add_vars_lcons(self, *args, **kwargs):
        return self._add_vars_lcons(*args, **kwargs)

    # methods to handle LnComb instance
    def add_lncomb(self, left, right=0):
        '''
            add constraint written as LnComb object
                left = right
                where both `left` and `right` are LnComb-like object
        '''
        assert isinstance(left, LnComb)

        p=left-right

        vs, ks, c=p.get_vs_coeffs_const()
        self._add_vars_lcons(vs, ks, -c)

    def repr_lncomb_with_basis(self, lncomb, return_lncomb=False):
        '''
            rerepsent a linear combination of variants
                in a basis

            return coeffs and const or LnComb instance

            Parameters:
                lncomb: LnComb-like object
                    linear combination to convert

                return_lncomb: bool
                    if True, return LnComb instance
                    otherwise, return tuple `(coeffs, const)`
        '''
        vs, ks, c=LnComb.lncomb_from(lncomb).get_vs_coeffs_const()
        assert all([self._contains(v) for v in vs])

        coeffs=_get_coeffs_container(self._get_len_basis())
        coeffs[:]=0

        const=c

        for v, k in zip(vs, ks):
            if self._is_const_var(v):
                c=self._get_const_var_val(v)
                const+=k*c
                continue

            if self._is_basis_var(v):
                i=self._get_basis_var_index(v)
                coeffs[i]+=k
                continue

            # lcomb
            ks, c=self._get_lcomb_var_coeffs_const(v)
            coeffs+=k*ks
            const+=k*c

        if return_lncomb:
            return LnComb(self._get_basis_vars_list(),
                            coeffs, const)

        return coeffs, const

    def eval_lncomb(self, lncomb):
        '''
            evaluate a linear combination

            if not a constant result, return None 
        '''
        ks, c=self.repr_lncomb_with_basis(lncomb, return_lncomb=False)
        if not self._all_zero(ks):
            return None
        return c

    def get_ratio_between_lncombs(self, t0, t1):
        '''
            return ratio of linear combinations, t0/t1

            if not constant ratio, return None
        '''
        ks0, c0=self.repr_lncomb_with_basis(t0, return_lncomb=False)
        ks1, c1=self.repr_lncomb_with_basis(t1, return_lncomb=False)

        indsnz0=self._inds_nonzero(ks0)
        indsnz1=self._inds_nonzero(ks1)
        indsnz=[*indsnz0, *indsnz1]

        if not indsnz:
            ratios=(c0, c1)
        else:
            i=indsnz[0]
            k0, k1=ks0[i], ks1[i]
            ratios=(k0, k1)

            zk=self._all_zero(k1*ks0-k0*ks1)
            zc=self._is_zero(k1*c0-k0*c1)

            # not ratio
            if not (zk and zc):
                return None

        return ratios[0]/ratios[1]

    # print
    def info(self):
        '''
            print equations in manager
        '''
        vbs=self._get_basis_vars_list()

        for vi in self._lcomb_vars:
            ks, c=self._get_lcomb_var_coeffs_const(vi)

            ks=[0 if self._is_zero(k) else k
                    for k in ks]

            print('%s = %s' % (vi, str(LnComb(vbs, ks, c))))

        for v, c in self._const_vars.items():
            if self._is_zero(c):
                c=0
            print('%s = %s' % (v, str(LnComb(const=c))))

        print('basis:', ', '.join(vbs))

class LnComb:
    '''
        class for linear combination with a bias
            k1*v1 + ... + kn*vn + c
    '''
    def __init__(self, vs=[], coeffs=1, const=0):
        '''
            init of linear combination class
        '''
        self._vars=list(vs)

        # const
        self._const=float(const)

        # coeffs
        n=len(self._vars)
        _coeffs=_get_coeffs_container(n)
        _coeffs[:]=_fill_coeffs_to_len(coeffs, n)

        self._coeffs=_coeffs

    def copy(self):
        return self.__class__(self._vars, self._coeffs, self._const)

    def get_vs_coeffs_const(self):
        return self._vars, self._coeffs, self._const

    # merge like terms
    def merge_like_terms(self):
        '''
            merge like terms with same variant
        '''
        vs=[]
        ks=[]

        vs0, ks0, c=self.get_vs_coeffs_const()
        for v, k in zip(vs0, ks0):
            if v not in vs:
                vs.append(v)
                ks.append(k)
            else:
                i=vs.index(v)
                ks[i]+=k

        return self.__class__(vs, ks, c)

    # construct from other object
    @classmethod
    def lncomb_from(cls, t):
        '''
            return LnComb instance constructed from `t`

            suport 3 types
                - LnComb
                    return itself

                - number
                    return a const term

                - object with method `to_lncomb`
                    return `t.to_lncomb()`
        '''
        if isinstance(t, cls):
            return t

        if isinstance(t, numbers.Number):
            return cls(const=t)

        if hasattr(t, 'to_lncomb'):
            t=t.to_lncomb()
            assert isinstance(t, cls), \
                    'unexpected type returned from `to_lncomb`: ' + \
                    type(t).__name__
            return t

        raise TypeError('unsupported type to convert to %s: %s'
                            % (cls.__name__, type(t).__name__))

    # arithmetic
    def __add__(self, p):
        '''
            return self + p

            p must be object which could be passed to `as_lncomb`
        '''
        p=self.lncomb_from(p)

        vs0, ks0, c0=self.get_vs_coeffs_const()
        vs1, ks1, c1=p.get_vs_coeffs_const()

        return self.__class__([*vs0, *vs1], [*ks0, *ks1], c0+c1)

    def __mul__(self, k):
        '''
            return self*k
            only support number for k

            self: k1*v1 + .. + kn*vn + c
            return (k*k1)*v1 + .. + (k*kn)*vn + (k*c)
        '''
        assert isinstance(k, numbers.Number), \
                'only support number type for k, ' \
                'but got %s' % type(k).__name__

        vs, ks, c=self.get_vs_coeffs_const()
        return self.__class__(vs, ks*k, c*k)

    ## other arithmetics based on `add` and `mul`
    def __pos__(self):
        # +self
        return self.copy()

    def __neg__(self):
        # -self
        return self.__mul__(-1)

    def __radd__(self, p):
        # p+self
        return self.lncomb_from(p).__add__(self)

    def __sub__(self, p):
        # self - p
        return self.__add__(self.lncomb_from(p).__neg__())

    def __rsub__(self, p):
        # p-self
        return self.lncomb_from(p).__sub__(self)

    def __rmul__(self, k):
        # k*self
        return self.__mul__(k)

    def __truediv__(self, k):
        # self/k
        return self.__mul__(1/k)

    # to string and print
    def _flt_fmt(self, k):
        '''
            formatter of float
        '''
        if int(k)==k:
            return str(int(k))
        return '%.4g' % k

    def _join_str_terms(self, terms):
        '''
            join string of terms
        '''
        s=''
        for t in terms:
            if not s or t.startswith('-'):
                s+=t
            else:
                s+='+'+t
        return s

    def __str__(self):
        vs, ks, c=self.get_vs_coeffs_const()

        terms=[]
        for k, v in zip(ks, vs):
            if k==0:
                continue

            if np.abs(k)!=1:
                sk=self._flt_fmt(k)
                terms.append(sk+'*'+v)
            elif k==1:
                terms.append(v)
            else:
                terms.append('-'+v)

        if c!=0 or not terms:
            terms.append(self._flt_fmt(c))

        return self._join_str_terms(terms)

    def __repr__(self):
        return '%s(%s, %s, %s)' % (self.__class__.__name__,
                                   repr(self._vars),
                                   repr(list(self._coeffs)),
                                   repr(self._const))

# auxiliary functions
def _fill_coeffs_to_len(coeffs, n):
    '''
        fill a given coeffs to array with len `n`

        4 type of `coeffs`
            - scalar: k
                return [k]*n
            - list-like with len < n
                fill 0 in end
            - list-like with len == n
                return itself
            - list-like with len > n
                raise Exception
    '''
    if isinstance(coeffs, numbers.Number):
        return [coeffs]*n

    nc=len(coeffs)
    if nc<n:
        coeffs=list(coeffs)+([0]*(n-nc))
    elif nc>n:
        raise Exception('unexpected len for coeffs: %i' % nc)

    return coeffs

def _get_coeffs_container(n):
    '''
        return container for coeffs with len `n`
    '''
    return np.full(n, np.nan, dtype='float64')

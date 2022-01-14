#!/usr/bin/env python3

'''
    base classes for linear combination

    a linear combination (with bias) has the form
        k1*v1+k2*v2+...+kn*vn+c
    where
        ks=[k1, k2, ..., kn]: coefficients
        c: const bias

    Classes:
        LnCoeffs
        LnComb

'''

import numbers
from collections import abc

import numpy as np

from .tools import insert_iter_to_list

# coeffs
class LnCoeffs:
    '''
        class for coefficients and bias of linear combinations,
        that is
            k1*v1+k2*v2+...+kn*vn+c
        for some variables, [v1, v2, ..., vn]

        In this class, 
            variables: ignored
            
            ks: stored in ndarray,
                facilitating subsequent numerical operation

            c: single attribute
    '''
    def __init__(self, ks, c, dtype=None, fixed_dtype=None):
        '''
            init work

            Parameters:
                ks: int or Iterable
                    if int, len of coeffs
                        only init buffer,
                            left values to fill later

                    if iter, must be able to
                        convert to 1-d array of numbers

                c: number
                    const bias

                dtype: numpy dtype, optional
                    dtype of array for ks

                    if not given, determined by
                        `np.array` or
                        `np.empty` (int `ks` given)

                fixed_dtype: bool or None
                    whether to raise dtype if value to set has higher type
                        for example, set float to int array

                    if None, False if `dtype` not given
                             True  otherwise
        '''
        # dtype fixed
        if fixed_dtype is None:
            fixed_dtype=False if dtype is None else True
        self._fixed_dtype=fixed_dtype

        # coeffs
        if isinstance(ks, abc.Iterable):
            ks=np.array(ks, dtype=dtype)
            if not self._is_valid_ks_array(ks):
                raise ValueError('invalid ks')

            self._ks=ks
            self._dtype=ks.dtype
        else:
            if not isinstance(ks, numbers.Integral):
                raise TypeError('only allow iterable or int for `ks`')

            n=ks
            self._init_ks_buffer(n, dtype=dtype)

        # const bias
        self.set_c(c)

    def _init_ks_buffer(self, n, dtype=None):
        '''
            init buffer for ks
        '''
        ks=self._new_ks_buffer(n, dtype)
        if not self._is_valid_ks_dtype(ks.dtype):
            raise TypeError('invalid dtype for ks')

        self._ks=ks
        self._dtype=ks.dtype

    def _reinit_ks_to_len(self, n):
        '''
            only change len of buffer
                leave buffer empty to fill
            keep dtype as previous
        '''
        assert isinstance(n, numbers.Integral)
        self._ks=self._new_ks_buffer(n, self._dtype)

    def _new_ks_buffer(self, n, dtype=None):
        '''
            new array for ks
        '''
        return np.empty(n, dtype=dtype)

    # dtype
    def to_dtype(self, dtype):
        '''
            change dtype of ks array
        '''
        if self._dtype==dtype:
            return

        ks=self._ks.astype(dtype)
        if not self._is_valid_ks_dtype(ks.dtype):
            raise TypeError('invalid dtype for ks')

        self._ks=ks
        self._dtype=ks.dtype

    def set_fixed_dtype(self, dtype=None):
        '''
            set ks dtype fixed

            :param dtype: dtype
                fixed dtype
                if None, use current dtype
        '''
        if dtype is not None:
            self.to_dtype(dtype)

        self._fixed_dtype=True

    def set_flex_dtype(self):
        '''
            set ks dtype flexible
        '''
        self._fixed_dtype=False

    def _raise_dtype_for_ks_set(self, v):
        '''
            raise dtype of ks
                if value to set has higher dtype
        '''
        if self._fixed_dtype:
            return

        v=np.asarray(v)
        if v.dtype>self._dtype:
            self.to_dtype(v.dtype)

    # copy
    def copy(self):
        '''
            copy as new instance
        '''
        return self.__class__(self._ks, self._c)

    # other constuctor
    @classmethod
    def lncoeffs(cls, p):
        '''
            construct instance from other object

            supported types:
                - LnCoeffs
                    return itself

                - tuple
                    (ks, c)
        '''
        if isinstance(p, cls):
            return p

        if type(p) is tuple:
            ks, c=p
            return cls(ks, c)

        raise TypeError('unsupported type to convert to %s: %s'
                            % (cls.__name__, type(p).__name__))

    # type check
    def _is_valid_ks_dtype(self, dtype):
        '''
            whether valid dtype for ks

            only allow dtype of number
        '''
        return issubclass(dtype.type, numbers.Number)

    def _is_valid_ks_array(self, array):
        '''
            whether valid for ks

            only allow 1-dimension array of numbers
        '''
        return array.ndim==1 and self._is_valid_ks_dtype(array.dtype)

    def _is_valid_c(self, c):
        '''
            valid type for const bias

            only allow number type
        '''
        return isinstance(c, numbers.Number)

    # setter
    def set_ks_items(self, inds, vs):
        '''
            set part of ks
            base method

            used like array set: array[inds]=vs
        '''
        vs=np.asarray(vs)
        self._raise_dtype_for_ks_set(vs)

        self._ks[inds]=vs

    def set_ks(self, ks):
        '''
            set coeffs, `ks`
        '''
        self.set_ks_items(slice(None), ks)

    def set_c(self, c):
        '''
            set bias, `c`
        '''
        assert self._is_valid_c(c), 'invalid type for `c`'
        self._c=c

    def set_ks_c(self, ks, c):
        '''
            set both ks and c
        '''
        self.set_ks(ks)
        self.set_c(c)

    # change len of ks
    def insert(self, ind, vals):
        '''
            insert list `vals` before index `ind`
        '''
        ks1=insert_iter_to_list(self._ks, ind, vals, inplace=False)

        self._reinit_ks_to_len(len(ks1))
        self.set_ks(ks1)

    def insert_zeros(self, ind, num):
        '''
            insert `num` 0 before index `ind`

            always used in new vars inserting
            `ks` would be filled with 0 in this position

            new `ks` would become, e.g. ind=i
                [k1, k2, ..., k_(i-1), 0, .., 0, ki, ...]
        '''
        self.insert(ind, [0]*num)

    def append(self, k):
        '''
            append new k to coeffs
        '''
        n=self.get_num_ks()
        self.insert(n, [k])

    def extend(self, ks):
        '''
            extend list of `ks` to current coeffs
        '''
        n=self.get_num_ks()
        self.insert(n, ks)

    def fill_to_len(self, n, fill=0):
        '''
            fill value to end of `ks`
                until length == `n`
        '''
        n0=self.get_num_ks()
        assert n0<=n, 'already exists too many `ks`'

        if n==n0:
            return

        self.insert(n0, [fill]*(n-n0))

    def pop(self, ind=-1):
        '''
            pop vars with index `ind`
            return corresponding coeff, `ki`
        '''
        ks1=list(self._ks)
        ki=ks1.pop(ind)

        self._reinit_ks_to_len(len(ks1))
        self.set_ks(ks1)

        return ki

    # arithmetic
    def inc_ks_by(self, ks):
        '''
            increase coeffs by `ks`
        '''
        if not isinstance(ks, numbers.Number):
            # if not number, must array-able
            ks=np.asarray(ks)
            assert len(ks)==len(self._ks) and ks.ndim==1,\
                    'mismatch len between `ks`'

        self.set_ks(self._ks+ks)

    def inc_c_by(self, c):
        '''
            increase const bias by `c`
        '''
        assert isinstance(c, numbers.Real), 'only allow inc by float'
        self._c+=c

    def inc_ks_c_by(self, ks, c):
        '''
            increase both ks and c
        '''
        self.inc_ks_by(ks)
        self.inc_c_by(c)

    def mul_ks_by(self, t):
        '''
            multiply `ks` by a factor `t`
        '''
        assert isinstance(t, numbers.Number), 'only allow mul by number'
        self.set_ks(self._ks*t)

    def mul_c_by(self, t):
        '''
            multiply `c` by a factor `t`
        '''
        assert isinstance(t, numbers.Number), 'only allow mul by number'
        self._c*=t

    def mul_ks_c_by(self, t):
        '''
            mutiply both `ks` and `c` by `t`
        '''
        self.mul_ks_by(t)
        self.mul_c_by(t)

    ## operator
    ### basic method
    def __neg__(self):
        # -self
        return self.__class__(-self._ks, -self._c)

    def __iadd__(self, p):
        # self+=p
        p=self.lncoeffs(p)
        self.inc_ks_c_by(p.coeffs, p.const)
        return self

    def __imul__(self, k):
        # self*=k
        self.mul_ks_c_by(k)
        return self

    def __add__(self, p):
        # self+p, different with `iadd` in dtype
        p=self.lncoeffs(p)
        return self.__class__(self._ks+p._ks, self._c+p._c)

    def __mul__(self, k):
        # self*k, different with `iadd` in dtype
        if not isinstance(k, numbers.Number):
            raise TypeError('only allow mul by number')
        return self.__class__(k*self._ks, k*self._c)

    ### derived from basic methods
    def __isub__(self, p):
        # self-=p
        return self.__iadd__(p.__neg__())

    def __sub__(self, p):
        # self-t
        return self.__add__(p.__neg__())

    def __truediv__(self, k):
        # self/k
        return self.__mul__(1/k)

    def __itruediv__(self, k):
        # self/=k
        return self.__imul__(1/k)

    ## operations in right
    def __rmul__(self, k):
        # k*self
        return self.__mul__(k)

    # getter
    def get_ks(self):
        '''
            return coeffs, `ks`
        '''
        return self._ks

    def get_c(self):
        '''
            return bias, `c`
        '''
        return self._c

    def get_ks_c(self):
        '''
            return tuple, (ks, c)
        '''
        return self.get_ks(), self.get_c()

    def get_num_ks(self):
        '''
            number of coeffs, `ks`
        '''
        return len(self._ks)

    ## properties
    ks=coeffs=property(get_ks, doc='coefficents')
    const=property(get_c, doc='const bias')

    ksc=property(get_ks_c, doc='(coeffs, const)')

    ## getitem and setitem, treated as `ks` array
    def __getitem__(self, i):
        '''
            getitem just to `ks` array

            only allow integral and slice as index
                - integral
                    return ks[i]
                - slice
                    return LnCoeffs(ks[slice], c)
        '''
        if isinstance(i, numbers.Integral):
            return self._ks[i]

        if isinstance(i, slice):
            return self.__class__(self._ks[i], self._c)

        raise IndexError('only allow integral or slice as index')

    def __setitem__(self, index, v):
        '''
            if v could be interpreted as (ks, c), like LnCoeffs
                set both ks and c
        '''
        if type(v) is LnCoeffs:
            assert isinstance(index, slice), \
                'only allow set LnCoeffs with slice index'
            ks, c=v
            self.set_ks_items(index, ks)
            self.set_c(c)
            return

        # otherwise, only set ks
        self.set_ks_items(index, v)

    ### declare not iter and not array
    def __iter__(self):
        '''
            declare iterable explicity
                to support syntax, e.g. ks, c=ksc

            if not, since __getitem__,
                it would be treated as iterable via index-based way
        '''
        return iter((self._ks, self._c))

    def __array__(self):
        '''
            handle numeric operation with ndarray
                especially 0-dimension
                    which may has different performance
                        with normal number type
            numeric op to ndarray, like a*p, always call
                a*asarray(p)

            return a 0-dimensin array
        '''
        # treat as 0-dimension array
        t=np.ndarray((), dtype=object)  # 0-dimension array
        t.fill(self)
        return t

    ## __len__
    def __len__(self):
        '''
            len of vars, that is num of coeffs
        '''
        return self.get_num_ks()

    # to string
    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__,
                               repr(list(self._ks)),
                               repr(self._c))

# combinations
class LnComb:
    '''
        class for linear combination with a bias
            k1*v1 + ... + kn*vn + c
    '''
    def __init__(self, vs=[], coeffs=1, const=0, dtype=None):
        '''
            init of linear combination class
        '''
        self._vars=list(vs)

        # coeffs and bias
        self._ksc=LnCoeffs(len(self._vars), const, dtype=dtype)
        self._ksc.set_ks(coeffs)

    def copy(self):
        return self.__class__(*self.get_vs_ks_c())

    def get_vs_ks_c(self):
        return self._vars, *self._ksc

    # merge like terms
    def merge_like_terms(self):
        '''
            merge like terms with same variant
        '''
        vs=[]
        ks=[]

        vs0, ks0, c=self.get_vs_ks_c()
        for v, k in zip(vs0, ks0):
            if v not in vs:
                vs.append(v)
                ks.append(k)
            else:
                i=vs.index(v)
                ks[i]+=k

        return self.__class__(vs, ks, c)

    # constant
    def is_const(self):
        t=self.merge_like_terms()
        if len(t._vars)==0:
            return True
        return False

    def asfloat(self):
        '''
            convert to float if const
        '''
        t=self.merge_like_terms()
        assert len(t._vars)==0, 'not constant LnComb'

        return t._ksc.const

    # construct from other object
    @classmethod
    def lncomb(cls, t):
        '''
            return LnComb instance constructed from `t`

            supported types:
                - LnComb
                    return itself

                - number
                    return a const term

                - list
                    interpreted as list of variables
                    use default coeffs 1 and const 0

                - tuple
                    (vs, [ks, [c]]) or (vs, [LnCoeffs])

                - object with method `to_lncomb`
                    return `t.to_lncomb()`
        '''
        if isinstance(t, cls):
            return t

        if isinstance(t, numbers.Number):
            return cls(const=t)

        if type(t) is list:
            return LnComb(t, 1, 0)

        if type(t) is tuple:
            if len(t)==2 and isinstance(t, LnCoeffs):
                vs, ksc=t
                assert len(vs)==len(ksc.coeffs), \
                    'missmatch between `vs` and `coeffs`'
                return LnComb(vs, ksc.coeffs, ksc.const)
            return LnComb(*t)

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
        p=self.lncomb(p)

        vs0, ks0, c0=self.get_vs_ks_c()
        vs1, ks1, c1=p.get_vs_ks_c()

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

        vs, ks, c=self.get_vs_ks_c()
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
        return self.lncomb(p).__add__(self)

    def __sub__(self, p):
        # self - p
        return self.__add__(self.lncomb(p).__neg__())

    def __rsub__(self, p):
        # p-self
        return self.lncomb(p).__sub__(self)

    def __rmul__(self, k):
        # k*self
        return self.__mul__(k)

    def __truediv__(self, k):
        # self/k
        return self.__mul__(1/k)

    ## to float if const
    def __float__(self):
        '''
            to float if const
        '''
        return self.asfloat()

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
        vs, ks, c=self.get_vs_ks_c()

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
                                   repr(list(self._ksc.coeffs)),
                                   repr(self._ksc.const))

# Exception class for linear
class LnConflictError(Exception):
    '''
        Error for conflict in linear constraints
    '''
    pass

#!/usr/bin/env python3

'''
    additional transform classes used in matplotlib
'''

import numpy as np

import matplotlib.transforms as mtrans

# wrapper of transform's inverse
class invTransWrapper(mtrans.Transform):
    '''
        handle inverse of transform
        mainly about invalidate from child, inverse of which to be computed
            in general, transform(values) = affine(non-affine(values)
            for normal transform, if affine invalid passed, only affect affine part,
                leaving non-affine unchanged.
            BUT for its inverse, the non affine should also change
    '''
    pass_through=True
    has_inverse=True

    def __init__(self, trans):
        super().__init__()

        self.input_dims=trans.output_dims
        self.output_dims=trans.input_dims
        # self.is_affine=trans.is_affine

        self._trans_init=trans
        self.set_children(trans)

    is_affine=property(lambda self: self._trans_init.is_affine)

    def _invalidate_internal(self, value, invalidating_node):
        '''
            handle invalidate for inverse
        '''
        if (value == self.INVALID_AFFINE and not self.is_affine):
            value=self.INVALID

        super()._invalidate_internal(value=value,
                                     invalidating_node=invalidating_node)

    def get_affine(self):
        if self.is_affine:
            return self._trans_init.inverted()
        return super().get_affine()

    def transform_non_affine(self, values):
        if self.is_affine:
            return values
        return self._trans_init.inverted().transform(values)

    def inverted(self):
        return self._trans_init

# perform func to xs given in axes coordinates
class yFuncTransFormFromAxes(mtrans.Transform):
    '''
        class to transform xs in axes coordinates
            to xys of a function's graph in an axes
    '''
    pass_through=True
    has_inverse=False

    def __init__(self, ax, yfunc):
        super().__init__()

        # set children to follow
        trans_d2a=ax.transScale+ax.transLimits
        self._transxa2d=invTransWrapper(trans_d2a)
        
        self._transdata=ax.transData

        self.set_children(self._transxa2d, self._transdata)

        # function
        assert callable(yfunc)
        self._yfunc=yfunc

        self.input_dims=self.output_dims=2

    def _invalidate_internal(self, value, invalidating_node):
        '''
            handle invalidate for inverse
        '''
        if (value == self.INVALID_AFFINE and
            invalidating_node is self._transxa2d):
            value=self.INVALID

        super()._invalidate_internal(value=value,
                                     invalidating_node=invalidating_node)

    def get_affine(self, *args):
        return self._transdata.get_affine()

    def transform_non_affine(self, values):
        # axes to data
        values=self._transxa2d.transform(values)

        # perform yfunc
        xs, _=values.T
        values=np.column_stack([xs, self._yfunc(xs)])

        # tranform data
        values=self._transdata.transform_non_affine(values)

        return values

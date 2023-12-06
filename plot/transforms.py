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

    def _invalidate_internal(self, level, invalidating_node):
        '''
            handle invalidate for inverse
        '''
        if (level == self.INVALID_AFFINE and not self.is_affine):
            level=self.INVALID

        super()._invalidate_internal(level=level,
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

    def _invalidate_internal(self, level, invalidating_node):
        '''
            handle invalidate for inverse
        '''
        if (level == self.INVALID_AFFINE and
            invalidating_node is self._transxa2d):
            level=self.INVALID

        super()._invalidate_internal(level=level,
                                     invalidating_node=invalidating_node)

    def get_affine(self, *args):
        return self._transdata.get_affine()

    def transform_non_affine(self, values):
        # axes to data
        values=self._transxa2d.transform(values)

        # perform yfunc
        xs=values[:, 0]
        values=np.column_stack([xs, self._yfunc(xs)])

        # tranform data
        values=self._transdata.transform_non_affine(values)

        return values

# composited scale transform
class CompositeAxisScaleTrans(mtrans.Transform):
    '''
        composite of scale transforms

        same scale in both axes as axis 'x' or 'y'
    '''
    pass_through=True

    def __init__(self, transforms, func, axis='x'):
        '''
            initiation

            Parameters:
                transforms: list of mtrans.Transform instances
                    tranforms to composite

                    for each transform, only scale in one axis used

                    any kind of transform allowed
                    no type check done
                        maybe weird result for non-affine and non-separable type

                    empty list also allowed
                        work like `Affine2D().scale(func())`

                func: callable
                    method to composite scales from transforms
                    any function is allowed

                    accept number of variables as that of `transforms`

                axis: 'x' or 'y', default 'x'
                    scale in which axis to composite
        '''
        super().__init__()

        self._trans=list(transforms)
        if self._trans:
            self.set_children(*self._trans)

        assert callable(func)
        self._func=func

        self._ind_axis=dict(x=0, y=1)[axis]

        self.input_dims=self.output_dims=2

    def get_scale(self):
        i=self._ind_axis
        scales=[t.get_matrix()[i, i] for t in self._trans]
        scale=self._func(*scales)
        # print(scale)
        return scale

    def get_affine(self, *args):
        scale=self.get_scale()
        return mtrans.Affine2D().scale(scale).get_affine()

# transforms of paths
def get_transforms_sr(angles=0, ratios=1, base_axis='x'):
    '''
        get transforms of scaling and rotating transform

        deformation only contains scaling, rotation
            no shearing
        and for scaling, area of path is kept
            that means only scaling (a, 1/a)

        Parameters:
            angles: array-like
                angles to rotate

            ratios: array-like
                ratio of another axis to base axis

            base_axis: 'x' 'y', default 'x'
                base axis for `ratios`
    '''
    angles, ratios, _=np.broadcast_arrays(angles, ratios, [0])
    assert base_axis in ['x', 'y']

    angles=angles*np.pi/180
    cosa=np.cos(angles)
    sina=np.sin(angles)

    # scale in x/y-axis with area of marker kept
    sy=np.sqrt(ratios)
    sx=1/sy
    if base_axis=='y':
        sx, sy=sy, sx

    transforms_sr=np.zeros((len(angles), 3, 3))
    transforms_sr[:, 0, :2]=np.column_stack([sx*cosa, -sy*sina])
    transforms_sr[:, 1, :2]=np.column_stack([sx*sina,  sy*cosa])
    transforms_sr[:, 2,  2]=1

    return  transforms_sr

def combine_paths_transforms(trans0, trans1):
    '''
        combine two transforms of path collection
        
        return transform
            which acts on paths like `trans1(trans0(paths))`
    '''
    trans0, trans1=np.broadcast_arrays(trans0, trans1)
    trans0=trans0.copy()
    for i in range(len(trans0)):
        trans0[i]=np.dot(trans1[i], trans0[i])

    return trans0

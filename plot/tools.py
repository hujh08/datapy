#!/usr/bin/env python3

'''
    useful tools for plot task
'''

from collections import abc
import numpy as np

## nested collection
### map function to nested collection
def is_scalar_default(v):
    '''
        return True if not `collections.abc.Collection`
    '''
    # return isinstance(v, numbers.Number)
    return not isinstance(v, abc.Collection)

def colection_astype_default(t_dst, data):
    '''
        convert collection of datat to a type `t_dst`
    '''
    if issubclass(t_dst, np.ndarray):
        return np.array(axes)

    if issubclass(t_dst, list) or \
       issubclass(t_dst, tuple):
        return t_dst(data)

    return data

def map_to_nested(f, elements,
                     is_scalar=is_scalar_default,
                     astype=colection_astype_default,
                     skip_none=False):
    '''
        map function to element in nested collection

        Parameters:
            is_scalar: callable
                determine whether an element is scalar

                default:
                    True for not `collections.abc.Collection`

            astype: None or callable
                if None, return nested of list

                default is to keep type of elements as possible

            skip_none: bool
                if True, skip None result

    '''
    if is_scalar(elements):
        return f(elements)

    result=[]
    for v in elements:
        t=map_to_nested(f, v, is_scalar, astype, skip_none)

        if skip_none and t is None:
            continue

        result.append(t)

    if astype is None:
        return result

    return astype(type(elements), result)

### squeeze to 1d array
def squeeze_nested(elements, is_scalar=is_scalar_default):
    '''
        squeeze nested collection to list
    '''
    if is_scalar(elements):
        return elements

    result=[]
    for v in elements:
        if is_scalar(v):
            result.append(v)
            continue

        result.extend(squeeze_nested(v, is_scalar))

    return result

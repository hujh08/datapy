#!/usr/bin/env python3

'''
    frequently used tools in pandas
'''

import numpy as np
import pandas as pd

# xs
def xs_by_dict(df, axis=0, drop_level=True, **kwargs):
    '''
        xs by dict

        xs in pandas: df.xs(key, axis, level)
        for multi-levels,
            df.xs([k0, k1], level=[l0, l1])

        Now support query by dict
            xs_by_dict(df, l0=k0, l1=k1)
    '''
    levels, labels=list(zip(*kwargs.items()))
    return df.xs(labels, level=levels, axis=axis, drop_level=drop_level)

# na
def has_na(d):
    '''
        whether has any NA value
    '''
    return np.any(pd.isna(d))

# to 2d table
def df_to_2dtab(df, xcol, ycol, vcol, fillna=None):
    '''
        convert df, like
            xcol  ycol vcol
            x0    y0   v00
            x0    y1   v01
            x1    y0   v10
            x1    y1   v11
        to 2d table, like
                x0    x1
            y0  v00   v10
            y1  v01   v11

        useful for df with product-type of index
    '''
    df=df.reset_index()[[xcol, ycol, vcol]]
    dt0=df[vcol].dtype

    dfs_xs=[]
    xs=[]
    for xi, dfi in df.groupby(xcol):
        xs.append(xi)
        dfs_xs.append(dfi.set_index(ycol)[vcol])

    # dftab=pd.concat(dfs_xs, keys=xs, names=[xcol], axis=1)
    dftab=pd.concat(dfs_xs, keys=xs, axis=1)
    if fillna is not None:
        dftab=dftab.fillna(fillna)

    dftab=dftab.astype(dt0).reset_index()

    return dftab
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

# loc
def df_loc_on_col(df, col, vals):
    '''
        loc df in a col

        select rows with val for `col` in `vals`
    '''
    return df.loc[df[col].isin(vals)]

# sort
def sort_index_by_list(df, klist, **kwargs):
    '''
        sort index as order in given list
    '''
    fkey=_list_to_sortkey(klist)
    return df.sort_index(key=fkey, **kwargs)

def sort_values_by_key(df, by, key=None, **kwargs):
    '''
        sort more flexible parameter for `key`,
            that is dict, list
    '''
    return df.sort_values(by, key=_norm_sort_key(key), **kwargs)

## auxiliary functions
def _list_to_sortkey(klist):
    kmap={}
    for i, t in enumerate(klist):
        assert t not in kmap  # no duplicate
        kmap[t]=i
    maxi=len(klist)  # max index for non-specified key
    return np.vectorize(lambda t: kmap.get(t, maxi))

def _norm_sort_key(key):
    '''
        normalize value for `key` in `sort_values_by_key`
    '''
    if isinstance(key, dict):
        fkeys={k: _norm_sort_key(v) for k, v in key.items()}
        def sortkey_dict(s):
            # input arg `s` is Series
            if hasattr(s, 'name') and s.name in fkeys:
                return fkeys[s.name](s)
            return s
        return sortkey_dict

    if isinstance(key, list):
        return _list_to_sortkey(key)

    return key

# print
def print_df(df, reset_index=False, **kwargs):
    '''
        more flexible function to print df
        use `df.to_string`
    '''
    if reset_index:
        df=df.reset_index()

    print(df.to_string(**kwargs))

# na
def has_na(d):
    '''
        whether has any NA value
    '''
    return np.any(pd.isna(d))

# to 2d table
def df_to_2dtab(df, xcol, ycol, vcol, fillna=None,
                    xkey=None, ykey=None,
                    keep_dtype=True,
                    drop_xname=True, reset_yind=False):
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
    if keep_dtype:
        dt0=df[vcol].dtype

    xs, dfs=[], []
    for xi, dfi in df.groupby(xcol):
        xs.append(xi)
        dfs.append(dfi.set_index(ycol)[vcol])

    kwargs=dict(keys=xs, axis=1)
    if not drop_xname:  # keep xname
        kwargs['names']=[xcol]
    dftab=pd.concat(dfs, **kwargs)

    # na
    if fillna is not None:
        dftab=dftab.fillna(fillna)

    # sort index
    if xkey is not None:
        dftab=dftab.sort_index(key=np.vectorize(xkey), axis=1)
    if ykey is not None:
        dftab=dftab.sort_index(key=np.vectorize(ykey), axis=0)

    # dtype
    if keep_dtype:
        dftab=dftab.astype(dt0)

    if reset_yind:
        dftab=dftab.reset_index()

    return dftab

# concat along index
def concat_dfs(dfs, mcol=None, marks=None, ignore_index=True):
    '''
        concat dataframe with new col to mark each
    '''
    if mcol is not None:
        assert len(dfs)==len(marks)

        dfs_new=[]
        for dfi, ni in zip(dfs, marks):
            dfi=dfi.copy()
            dfi[mcol]=ni
            dfs_new.append(dfi)
        dfs=dfs_new

    return pd.concat(dfs, ignore_index=ignore_index)

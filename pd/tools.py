#!/usr/bin/env python3

'''
    frequently used tools in pandas
'''

import collections

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

def sort_values_by_key(df, by, key=None, vectorized=False, **kwargs):
    '''
        sort values by given key

        :param key: callable, dict, Sequence, 1d np.ndarray
            if callable and not `vectorized`,
                it will apply to each element in Index, instead whole object
    '''
    key=_norm_sort_key(key, vectorized=vectorized)
    return df.sort_values(by, key=_norm_sort_key(key), **kwargs)

def sort_index_by_key(df, key, vectorized=False, **kwargs):
    '''
        sort index by given key

        :param key: callable, dict, Sequence, 1d np.ndarray
            if callable and not `vectorized`,
                it will apply to each element in Index, instead whole object
    '''
    key=_norm_sort_key(key, vectorized=vectorized)
    return df.sort_index(key=np.vectorize(key), **kwargs)

## auxiliary functions
def _list_to_sortkey(klist):
    kmap={}
    for i, t in enumerate(klist):
        assert t not in kmap  # no duplicate
        kmap[t]=i
    maxi=len(klist)  # max index for non-specified key
    return np.vectorize(lambda t: kmap.get(t, maxi))

def _norm_sort_key(key, vectorized=False):
    '''
        normalize value for `key` in `sort_values_by_key`

        :param key: callable, dict, Sequence, 1d np.ndarray
            if callable and not `vectorized`,
                it will apply to each element in Index, instead whole object
    '''
    if isinstance(key, dict):
        fkeys={k: _norm_sort_key(v) for k, v in key.items()}
        def sortkey_dict(s):
            # input arg `s` is Series
            if hasattr(s, 'name') and s.name in fkeys:
                return fkeys[s.name](s)
            return s
        return sortkey_dict

    if isinstance(key, collections.abc.Sequence):
        return _list_to_sortkey(key)

    if isinstance(key, np.ndarray):
        if key.ndim!=1:
            raise ValueError('only allow 1d `np.ndarray` as `key`')
        return _list_to_sortkey(key)

    if not callable(key):
        s='only allow callable, dict, Sequence, or 1d ndarray as `key`'
        raise ValueError(s)

    if not vectorized:
        return np.vectorize(key)

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

def print_tab(tab, reset_index=True, index=False, **kwargs):
    '''
        print 2d tab

        wrapper of `print_df`
    '''
    print_df(tab, reset_index=reset_index,
                  index=index, **kwargs)

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

        :param xkey, ykey: None, or callable, dict, Sequence, 1d np.ndarray
            key to sort xcol/ycol

            see `sort_index_by_key` for detail
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
        dftab=sort_index_by_key(dftab, key=xkey, axis=1)
    if ykey is not None:
        dftab=sort_index_by_key(dftab, key=ykey, axis=0)

    # dtype
    if keep_dtype:
        dftab=dftab.astype(dt0)

    if reset_yind:
        dftab=dftab.reset_index()

    return dftab

# statistic of df
def df_count_by_group(df, hcol, vcol, fillna=0,
                         sort_hcol=None, sort_vcol=None,
                         colname_count='_count', **kwargs):
    '''
        count df in group of two columns
        output a 2d table

        Parameters:
            df: dataframe
                data to count

            hcol: str
                name for horizontal axis of output table

            vcol: str
                name for vertical axis of output table

            sort_hcol, sort_vcol: None, or callable, dict, Sequence, 1d np.ndarray
                key to sort hcol or vcol

                see `sort_index_by_key` for detail

            colname_count: str
                column name for count result, use '_count' by default
                    must be different with `hcol` and `vcol`

                not affect result table
    '''
    df_cnt=df.groupby([hcol, vcol])\
             .size()\
             .to_frame(name=colname_count)\
             .reset_index()

    return df_to_2dtab(df_cnt, hcol, vcol, colname_count, fillna=fillna,
                        xkey=sort_hcol, ykey=sort_vcol, **kwargs)

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

# merge multi dfs
def merge_dfs(dfs, suffixes=None, on=None, **kwargs):
    '''
        merge multiply dataframes

        :param suffixes: None, or list of string
            suffixes to distinguish column

            if None, use default ['_1', '_2', ..., '_n']
    '''
    n=len(dfs)
    assert n>=2, 'only support merge >=2 dfs'

    # suffixes
    if suffixes is None:
        suffixes=[f'_{i+1}' for i in range(n)]
    else:
        assert len(suffixes)==n

    # rename columns
    cnt_colname={}
    for df in dfs:
        for c in df.columns.unique():
            if c not in cnt_colname:
                cnt_colname[c]=0
            cnt_colname[c]+=1

    ## exclude key in `on`
    if on is not None:
        excludes=[]
        if isinstance(on, str):
            excludes=[on]
        else:
            excludes=on

        for o in excludes:
            if o in cnt_colname:
                del cnt_colname[o]

    ## extract duplicated columns
    cols_dup=[s for s, v in cnt_colname.items() if v>=2]

    ## rename
    dfs1=[]
    for df, s in zip(dfs, suffixes):
        mapname={c: c+s for c in cols_dup}
        dfs1.append(df.rename(columns=mapname))
    dfs=dfs1

    # merge
    df, s=dfs[0], suffixes[0]
    for dfi, si in zip(dfs[1:], suffixes[1:]):
        df=df.merge(dfi, suffixes=[s, si], on=on, **kwargs)

    return df

def merge_dfs_on_ind(dfs, **kwargs):
    '''
        merge dataframes on index
    '''
    return merge_dfs(dfs, left_index=True, right_index=True, **kwargs)

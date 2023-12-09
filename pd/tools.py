#!/usr/bin/env python3

'''
    frequently used tools in pandas
'''

import collections

import numpy as np
import pandas as pd

__all__=['xs_by_dict', 'df_loc_on_col',
         'sort_index_by_list', 'sort_values_by_key', 'sort_index_by_key',
         'mv_cols_head',
         'print_df', 'print_tab',
         'has_na',
         'df_to_2dtab', 'df_count_by_group',
         'concat_dfs', 'merge_dfs', 'merge_dfs_on_ind']

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
    return df.sort_values(by, key=key, **kwargs)

def sort_index_by_key(df, key, vectorized=False, **kwargs):
    '''
        sort index by given key

        :param key: callable, dict, Sequence, 1d np.ndarray
            if callable and not `vectorized`,
                it will apply to each element in Index, instead whole object
    '''
    key=_norm_sort_key(key, vectorized=vectorized)
    return df.sort_index(key=key, **kwargs)

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

# adopt columns order
def mv_cols_head(df, cols):
    '''
        move given column to head
            ignore non-existed column
    '''
    assert df.columns.is_unique, 'non-unique column names'
    
    cols_df=list(df.columns)

    head=[]
    for c in cols:
        if c not in cols_df:
            continue

        cols_df.remove(c)
        head.append(c)

    return df[head+cols_df]

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
def df_to_2dtab(df, xcol, ycol, vcol,
                    aggfunc=None,
                    fillna=None,
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

        wrapper of `DataFrame.pivot` or `DataFrame.pivot_table`
            `xcol, ycol, vcol` for `index, columns, values`
        support multi-level name for `xcol`, `ycol`

        Parameters:
            xkey, ykey: None, or callable, dict, Sequence, 1d np.ndarray
                key to sort xcol/ycol

                see `sort_index_by_key` for detail

            keep_dtype: bool, default True
                whether to keep dtype as initial `vcol`

                dtype might change when na exists after pivot
    '''
    if keep_dtype:
        dt0=df[vcol].dtype

    # pivot
    func_piv=lambda df, **kws: df.pivot(**kws)
    kws_pv=dict(index=ycol, columns=xcol, values=vcol)
    if aggfunc is not None:
        func_piv=lambda df, **kws: df.pivot_table(**kws)
        kws_pv['aggfunc']=aggfunc

    dftab=func_piv(df, **kws_pv)

    if drop_xname:
        dftab.columns.name=None

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

    # reset index
    if reset_yind:
        dftab=dftab.reset_index()

    return dftab

# statistic of df
def df_count_by_group(df, hcol, vcol, fillna=0,
                        normalize=False, sort_cnt=False,
                        dropna_cnt=True,
                        sort_hcol=None, sort_vcol=None, **kwargs):
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

            normalize, sort_cnt, dropna_cnt:
                args for `df.value_counts`

                effect of `sort_cnt` may be overrided by `sort_hcol` or `sort_vcol`
                    if some not None
    '''
    subset=[]
    for col in [hcol, vcol]:
        if isinstance(col, str):
            col=[col]
        subset.extend(col)

    name_dfcnt='_count'
    if name_dfcnt in subset:
        for i in range(len(subset)+1):
            name_dfcnt=f'_count{i}'

    # value counts
    df_cnt=df.value_counts(subset=subset,
                           normalize=normalize, sort=sort_cnt,
                           dropna=dropna_cnt, ascending=False)\
             .to_frame(name=name_dfcnt)\
             .reset_index()

    # to table
    return df_to_2dtab(df_cnt, hcol, vcol, name_dfcnt, fillna=fillna,
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

#!/usr/bin/env python3

'''
    io of text file
'''

import numbers

import pandas as pd

from .utils import (lstrip_line_comment_chars, rstrip_line_comment,
                    read_nth_line)

__all__=['load_txt', 'load_txts', 'save_to_txt']

# read text
def load_txt(fileobj, line_nrow=None, header_comment=False,
                delim_whitespace=True, comment='#', 
                fields=None, map_to_srccols=None,
                **kwargs):
    '''
        load text file

        wrap of `pd.read_csv`
            customized for frequently-used style

        Parameters
            fileobj: str, path object or file-like object
                specify input file

                str or path object: refer to location of the file

                file-like object: like file handle or `StringIO`

            line_nrow: None or int
                line to specify number of rows to read

                if None,
                    no such line exists

            header_comment: bool
                specify whether comment char added before header line

                mark header in order to mask it in simple treatment of the file
                    like through tools `sed` or `awk`

                if `header_comment` is True and no `header` given
                    by default, use 0 or `line_nrow`+1 if it given

                    NOTE:
                        only support single header line
                            and whitespace separation in head line

            fields: list-like or callable, optional
                columns to output,

                in most cases, it is just alias of `usecols`
                    also columns in output
                if `map_to_srccols` given (and `fields` is list-like),
                    column names in source text file are different

            map_to_srccols: dict or None
                map name in `fields` to one in source text file

                only work for list-like `fields`
                    recommand to use `df.rename` after loading
                        if callable `fields` given
    '''
    skiprows=set()   #  for nrows and header line
    if line_nrow is not None:  # if given, fetch it
        assert 'nrows' not in kwargs  # conflict keyword

        line=read_nth_line(fileobj, line_nrow, restore_stream=True)

        if comment is not None:
            line=rstrip_line_comment(line, comment=comment)

        # nrows
        nrows=int(line)

        # update arguments
        kwargs['nrows']=nrows
        skiprows.add(line_nrow)

    if header_comment and comment is not None:
        assert 'names' not in kwargs  # conflict keyword

        n=line_nrow+1 if line_nrow is not None else 0  # default header
        header=kwargs.pop('header', n)
        if not isinstance(header, numbers.Integral):
            s='only support sinle header line'
            raise Exception(s)

        line=read_nth_line(fileobj, header, restore_stream=True)

        # remove head comment chars and end comment string
        assert len(comment)==1
        line=lstrip_line_comment_chars(line, comment)
        line=rstrip_line_comment(line, comment)

        # update arguments
        kwargs['names']=line.split()  # only whitespace separation
        skiprows.add(header)

    # update 'skiprows' in kwargs
    if skiprows:
        # skiprows
        if 'skiprows' not in kwargs:
            kwargs['skiprows']=skiprows
        else:
            t=kwargs['skiprows']

            if callable(t):
                kwargs['skiprows']=\
                    lambda x, f0=t, s0=skiprows: x in s0 or f0(x)
            else:
                if isinstance(t, numbers.Integral):
                    t=list(range(t))
                kwargs['skiprows']=skiprows.union(list(t))

    # fields, map to columns in source text
    if map_to_srccols is not None:
        assert isinstance(map_to_srccols, dict)
        assert hasattr(fields, '__iter__') # only work for list-like `fields`
        fields=[map_to_srccols.get(t, t) for t in fields]

    # alias of keywords
    if fields is not None:
        assert 'usecols' not in kwargs  # avoid conflict

        kwargs['usecols']=fields

    # load text through `pd.read_csv`
    kws1=dict(comment=comment)
    if delim_whitespace:
        kws1['sep']=r'\s+'
    df=pd.read_csv(fileobj, **kws1, **kwargs)

    # resume name given in `fields`
    if map_to_srccols is not None:
        rename={v: k for k, v in map_to_srccols.items()}
        df=df.rename(columns=rename)

    return df

def load_txts(files, ignore_index=True, **kwargs):
    '''
        load multiply txt files
        return concatenated dataframe
    '''
    dfs=[load_txt(f, **kwargs) for f in files]
    return pd.concat(dfs, ignore_index=ignore_index)

# write text
def save_to_txt(df, path_or_buf=None, index=False,
                    sep=' ', na_rep='NaN', **kwargs):
    '''
        save DataFrame to text file
            hujh-friendly default arguments

        wrap of method `DataFrame.to_string`

        changelog:
            2022/04/26: use `df.to_csv`, instead of `df.to_string`
                output of the latter has no '\\n' in last line
                    which may raise wrong result
                        for some frequently used routine
                            like `wc -l`

        Parameters:
            df: DataFrame
                object to save

            path_or_buf: str, Path or StringIO-like, or default None
                buffer to write to. same as `to_string`
                if None, output is returned
    '''
    kwargs.update(index=index, sep=sep, na_rep=na_rep)
    return df.to_csv(path_or_buf, **kwargs)

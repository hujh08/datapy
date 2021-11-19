#!/usr/bin/env python3

'''
    io of text file
'''

import numbers
import re

import pandas as pd

# read text
def load_txt(fileobj, line_nrow=None, header_comment=False,
                delim_whitespace=True, comment='#', 
                fields=None, **kwargs):
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

            header_comment: bool
                specify whether comment char added before header line

                mark header in order to mask it in simple treatment of the file
                    like through tools `sed` or `awk`

                if `header_comment` is True and no `header` given
                    by default, use 0, `line_nrow`+1 if it given

                    NOTE:
                        only support single header line
                            and whitespace separation in head line

            fields: list-like or callable, optional
                columns to extract,
                    alias of `usecols`
    '''
    skiprows=set()   #  for nrows and header line
    if line_nrow is not None:  # if given, fetch it
        assert 'nrows' not in kwargs  # conflict keyword

        line=read_nth_line(fileobj, line_nrow, restore_stream=True)

        if comment is not None:
            line=line_comment_strip(line, comment=comment)

        # nrows
        nrows=int(line)

        # update arguments
        kwargs['nrows']=nrows
        skiprows.add(line_nrow)

    if header_comment and comment is not None:
        assert 'names' not in kwargs  # conflict keyword

        n=line_nrow+1 if line_nrow is not None else 0  # default header
        header=kwargs.pop('header', n)
        assert isinstance(header, numbers.Integral)  # only support sinle header line

        line=read_nth_line(fileobj, header, restore_stream=True)

        # remove comment char in left
        assert len(comment)==1
        line=re.sub(r'^[%s\s]*' % comment, '', line)

        line=line_comment_strip(line, comment)

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

    # alias of keywords
    if fields is not None:
        assert 'usecols' not in kwargs  # avoid conflict

        kwargs['usecols']=fields

    # load text through `pd.read_csv`
    return pd.read_csv(fileobj, delim_whitespace=delim_whitespace,
                comment=comment, **kwargs)

## auxiliary functions
def read_nth_line(fileobj, n, restore_stream=False):
    '''
        read nth line

        `n` is 0-indexed

        `restore_stream`: bool
            whether to restore stream position

            if False, after return, current position would be (n+1)-th line
    '''
    if isinstance(fileobj, str):
        with open(fileobj) as f:
            return read_nth_line(f, n)

    assert n>=0
    if restore_stream:
        t=fileobj.tell()

    for _ in range(n+1):
        line=fileobj.readline()

    if restore_stream:
        fileobj.seek(t)

    return line

### remove comment
def line_comment_strip(line, comment='#'):
    '''
        remove comment in a line
    '''
    assert len(comment)==1  # only support single char

    return re.sub(r'[%s].*$' % comment, '', line)

# write text
def save_to_txt(df, buf=None, index=False, **kwargs):
    '''
        save DataFrame to text file

        wrap of method `DataFrame.to_string`

        Parameters:
            df: DataFrame
                object to save

            buf: str, Path or StringIO-like, or default None
                buffer to write to. same as `to_string`
                if None, output is returned
    '''
    return df.to_string(buf, index=index, **kwargs)

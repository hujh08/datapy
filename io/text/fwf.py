#!/usr/bin/env python3

'''
    io of fixed-width formatted text file
'''

import numbers

import pandas as pd

from .utils import strip_empty_lines, words_in_line

__all__=['load_fwf', 'load_fwf_by_descrip']

# load fwf
def load_fwf(file_or_buffer, colspec='infer', header=None,
                             **kwargs):
    '''
        read fixed-width formatted text

        wrap of `pd.read_fwf`
            except some default parameters reset

        default parameters:
            colspec: 'infer'
                see `pd.read_fwf` for details

            header: None
                no header specified
    '''
    kwargs.update(colspec=colspec, header=header)
    return pd.read_fwf(file_or_buffer, **kwargs)

# load fwf with readme doc
def load_fwf_by_descrip(file_or_buffer, descr,
                            how_descr=None,
                            col_bytes_in_descr=0,
                            col_label_in_descr=None, **kwargs):
    '''
        load fwf by description of the format

        Example of fwf description
            ----------------------------------------------------
               Bytes Format Units   Label     Description
            ----------------------------------------------------
               1- 18  A18   ---     name      object id
              22- 26  F5.2  unit    prop      object property
                  28  A1    ---     flag      object flag
            ----------------------------------------------------
        where
            first row, 'Bytes Format ...':
                header of description
            first column, '1- 18', '22- 26', ...:
                column specification of fwf in bytes form

        Parameters:
            how_descr: None, str, list
                how to parse description

                if None,
                    by simple row split

                if str,
                    by aligned header

                if list,
                    by col spec or col sep
    '''
    # bytes and label from descr
    if how_descr is None:
        f_parse=parse_fwf_descrip_by_split
        args=()
    else:
        args=(how_descr,)
        if isinstance(how_descr, str):
            f_parse=parse_fwf_descrip_by_header
        elif isinstance(how_descr[0], numbers.Number):
            f_parse=parse_fwf_descrip_by_colseps
        else:
            f_parse=parse_fwf_descrip_by_colspecs

    args_col=(col_bytes_in_descr, col_label_in_descr)
    cbytes_label=f_parse(descr, *args, *args_col)

    # load fwf
    if col_label_in_descr is None:
        kws_descr=dict(colspec=cbytes_label)
    else:
        cbytes, label=cbytes_label
        kws_descr=dict(colspec=cbytes, names=label)

    return load_fwf(file_or_buffer, **kws_descr, **kwargs)

## parse description in readme doc
def _parse_fwf_descrip(descrip, line_parser):
    '''
        base function to parse fwf description
    '''
    result_lines=[]

    lines=strip_empty_lines(descrip, join_lines=False)
    for line in lines:
        result_line=[]
        result_lines.append(result_line)

        fields=line_parser(line)

        # fwf colspec
        s_bytes=fields[0]
        result_line.append(parse_bytes_str(s_bytes))

        # fwf label
        if len(fields)<=1:
            continue
        s_label=fields[1]
        result_line.append(s_label.strip())

    result_fields=tuple(zip(*result_lines))

    if len(result_fields)<=1:
        return result_fields[0]

    return result_fields

### parse by simple split
def parse_fwf_descrip_by_split(descrip, ind_bytes=0, ind_label=None):
    '''
        parse description of fixed-width formatted text
            by simple split
    '''
    # line parser
    inds_ext=[ind_bytes]
    if ind_label is not None:
        inds_ext.append(ind_label)

    def line_parser(line):
        fields=line.split()
        return [fields[k] for k in inds_ext]

    # parse lines
    return _parse_fwf_descrip(descrip, line_parser)

### parse description by column specifications or separations
def parse_fwf_descrip_by_colspecs(descrip, colspec_bytes,
                                        colspec_label=None):
    '''
        parse description of fixed-width formatted text
            by given column specifications

        Return
            (colspecs, labels) or colspecs

        Example of description:
               1- 18  A18   ---     name      object id
              22- 26  F5.2  unit    prop      object property
                  28  A1    ---     flag      object flag
        colspecs and labels returned:
            colspecs: [(0, 18), (21, 26), (27, 28)]
            labels: ['name', 'prop', 'flag']

        Parameter:
            descrip: str
                description of fwf

            colspec_bytes: tuple or slice
                col spec for bytes field
                    i.e. 1st column of example

                bytes range of column in fwf,
                    indexing from 0

            colspec_label: None, tuple or slice, default None
                col spec for label field
                    i.e. 4th column of example

                name of columns in fwf file

                if None,
                    no name specified

                    return only `colspecs`
    '''
    # slices of field
    if not isinstance(colspec_bytes, slice):
        colspec_bytes=slice(*colspec_bytes)
    f_bytes=lambda line: line[colspec_bytes]

    if colspec_label is not None:
        if not isinstance(colspec_label, slice):
            colspec_label=slice(*colspec_label)
        f_label=lambda line: line[colspec_label]

        line_parser=lambda line: (f_bytes(line), f_label(line))
    else:
        line_parser=lambda line: (f_bytes(line),)

    # parse lines
    return _parse_fwf_descrip(descrip, line_parser)

def parse_fwf_descrip_by_colseps(descrip, seps, ind_bytes=0, ind_label=None):
    '''
        parse description of fixed-width formatted text
            by given column separations

        each separation is
            column index after last character of field,
                starting from 0

        no include that of last field
    '''
    seps=[None, *seps, None]
    pairs_sep=list(zip(seps[:-1], seps[1:]))

    colspec_bytes=pairs_sep[ind_bytes]

    colspec_label=None
    if ind_label is not None:
        colspec_label=pairs_sep[ind_label]

    return parse_fwf_descrip_by_colspecs(descrip,
                colspec_bytes=colspec_bytes,
                colspec_label=colspec_label)

### parse by aligned header
def parse_aligned_header(header):
    '''
        parse header
            aligned in first character of each field
                with body rows

        return
            (colspecs, fields)
    '''
    fields, tspans=words_in_line(header)

    starts=[t[0] for t in tspans[1:]]
    seps=[None, *starts, None]

    colspecs=list(zip(seps[:-1], seps[1:]))

    return colspecs, fields

def parse_fwf_descrip_by_header(descrip, header, 
                                    col_bytes=0, col_label=None):
    '''
        parse description of fixed-width formatted text
            by given aligned header

        Parameter:
            col_bytes: int, or str
                column for bytes

                if int,
                    column index
                if str,
                    name in `header`

            col_label: None, int, or str
                same as `col_bytes`

                if None,
                    no label column specified
    '''
    # parse aligned header
    cspecs, fields=parse_aligned_header(header)

    # col spec
    if not isinstance(col_bytes, numbers.Number):
        col_bytes=fields.index(col_bytes)
    colspec_bytes=cspecs[col_bytes]

    if col_label is None:
        colspec_label=None
    else:
        if not isinstance(col_label, numbers.Number):
            col_label=fields.index(col_label)
        colspec_label=cspecs[col_label]

    return parse_fwf_descrip_by_colspecs(descrip,
                colspec_bytes=colspec_bytes,
                colspec_label=colspec_label)

### auxiliary functions
def parse_bytes_str(sbytes, sep='-'):
    '''
        parse bytes string for colspec of fwf file
        
        sbytes is by column number, starting from 1
        colspec returned is index colspec, starting from 0

        for example:
            1 - 10  ==> (0, 10)
            10      ==> (9, 10)

        Parameters:
            sbytes: str
                string for bytes in fwf file

                e.g. '1-10', '10'

            sep: str, default '-'
                separator for `sbytes`
    '''
    byts=list(map(int, sbytes.split(sep)))
    if len(byts)<=1:
        t0=t1=byts[0]
    else:
        t0, t1=byts

    return t0-1, t1

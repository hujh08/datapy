#!/usr/bin/env python3

'''
    utilities for argument parser of YAML io
'''

# delimiter, for key-val string or multi-level key name
#     e.g. key=val, key0/key1/key2

def type_delim(s=None):
    # type of delim: return split func
    class Spliter:
        def __init__(self, d=None):
            self._delim=d

        @property
        def delim(self):
            if self._delim is None:
                return ' '
            return self._delim

        def __call__(self, t, maxsplit=-1):
            # return value as is when not str type
            if not isinstance(t, str):
                return t

            # split for str
            return t.split(self._delim, maxsplit=maxsplit)

        def join(self, vals, with_prefix=False):
            d=self.delim
            s=d.join(vals)
            if with_prefix:
                s=d+s

            return s

        def __str__(self):
            return self.delim

    return Spliter(s)

def add_delim_to_parser(parser, *opts, func_type=True, delim_of=None, **kws):
    if func_type:
        kws['type']=type_delim
        if kws.get('default', None) is None:
            kws['default']=type_delim(None)

    kws.setdefault('metavar', 'c')  # metavar

    # help
    if delim_of is not None:
        h=f"delimiter of {delim_of}. default `%(default)s`"
        kws.setdefault('help', h)  # help

    parser.add_argument(*opts, **kws)

def add_key_delim_to_parser(parser, default='/', **kws):
    '''
        delimeter for multi-level key
            e.g. grp/subgrp/name
    '''
    # forced kws
    kws0=dict(func_type=True, dest='keySplit', default=default,
                delim_of='multi-level key')

    add_delim_to_parser(parser, '-d', '--delimiter', **kws0, **kws)

def add_val_delim_to_parser(parser, default=',', **kws):
    '''
        delimeter for list val
            e.g. key='1,2,3'
    '''
    # forced kws
    kws0=dict(func_type=True, dest='valSplit', default=default,
                delim_of='list val')

    add_delim_to_parser(parser, '--delim-val', **kws0, **kws)

def add_keyval_delim_to_parser(parser, default='=', **kws):
    '''
        delimeter for keyval string
            e.g. key=val
    '''
    # forced kws
    kws0=dict(func_type=True, dest='kvSplit', default=default,
                delim_of='key-val str')

    add_delim_to_parser(parser, '-D', '--delimiter-kv', **kws0, **kws)

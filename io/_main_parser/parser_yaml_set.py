#!/usr/bin/env python3

'''
    parser of YAML set
'''

import os, shutil

from ..yaml import load_yaml, save_to_yaml

from .util import (new_parser, grpdata_get, grpdata_set)
from . import util_yaml

# construct parser
def create_parser(parent=None):
    kws=dict(parent=parent, help='YAML set',
                description='set keys for YAML')
    parser=new_parser('set', **kws)

    parser.add_argument('file', help='YAML file to set')
    parser.add_argument('keyvals', help=('key and val to set, '
                                         '`key=val`'),
                            nargs='+')

    parser.add_argument('-v', '--verbose', help='verbose of what is done',
                            action='store_true')

    util_yaml.add_keyval_delim_to_parser(parser)  # key-val delim
    util_yaml.add_key_delim_to_parser(parser)   # key delim
    util_yaml.add_val_delim_to_parser(parser)   # val delim

    parser.add_argument('-T', '--val-types',
                        # default: str if nothing specified for a var
                        # i,f: int, float
                        # prefix l: list, split by ','
                        # list of scalar type: e.g. 'ifs'
                        help=('type of values, '
                              '`key = i|f|l|li|lf|eval|list-scalar '
                              '(e.g. sif)`'),
                        nargs='+', dest='valtypes', default=[])

    # add func
    parser.set_defaults(func=func_of_parser)

    return parser

# function to handle args
def func_of_parser(args):
    if args.verbose:
        print(f'set `{args.file}`')

    if os.path.exists(args.file):
        data=load_yaml(args.file)
    else:
        if args.verbose:
            print('file not exists, create new one')
        data={}

    # parse val types
    map_valtypes={}
    for skt in args.valtypes:
        sk, t=args.kvSplit(skt)
        if sk in map_valtypes:
            raise ValueError(f'duplicated val type for key: `{sk}`')
        map_valtypes[sk]=parse_val_type(t, vspliter=args.valSplit)

    # set vals
    for skv in args.keyvals:
        sk, v=args.kvSplit(skv, maxsplit=1)
        if sk in map_valtypes:
            v=map_valtypes[sk](v)

        if args.verbose:
            print(f'set key-val: `{sk}`, `{repr(v)}`')

        k=args.keySplit(sk)
        grpdata_set(data, k, v)

    # save to dest file
    dst=args.file
    if args.verbose:
        print(f'write back to: {dst}')

    save_to_yaml(data, dst, safe_backup=True)

## parse val type
def parse_val_type(vtype, vspliter=None):
    funcs_default={
        'i': int,
        'f': float,
        'l': lambda v: [t.strip() for t in vspliter(v)],
        'li': lambda v: [int(t) for t in vspliter(v)],
        'lf': lambda v: [float(t) for t in vspliter(v)],
        'eval': eval
    }

    if vtype in funcs_default:
        func_type=funcs_default[vtype]
    else:  # list of scalar
        func_type=listvalTyper(vtype, vspliter)

    return func_type

class listvalTyper:
    '''
        type converter to list value
    '''
    def __init__(self, vtype, vspliter, default=None):
        self._list_funcs, self._index_extend=\
            self._parse_vtype(vtype)

        self._vspliter=vspliter

        if default is None:
            default=lambda v: v
        self._default=default

    @staticmethod
    def _parse_vtype(vtype):
        '''
            parse type for list of scalar

            :param vtype: str, consist of char 's', 'i', 'f', '*'
                's', 'i', 'f': str, int , float
                *: global match

                e.g.
                    'sif': str, int, float for exact 3 elements
                    '*sif': types for tail 3rd elements
                    'sif*': same as 'sif'
                    'si*f': types for two end points
        '''
        map_func_scalar={'i': int, 'f': float, 's': str}

        # extend types
        list_types=list(vtype)
        index_extend=None
        if '*' in list_types:
            index_extend=list_types.index('*')
            list_types.pop(index_extend)

        # check types
        if any(map(lambda t: t not in map_func_scalar, list_types)):
            s=f'got unexpected type for list of scalar: {vtype}'
            raise ValueError(s)

        list_funcs=[map_func_scalar[k] for k in list_types]

        return list_funcs, index_extend

    # function to list
    def _convert_list_vals(self, vals):
        lfs=list(self._list_funcs)

        if len(vals)<len(lfs):
            raise ValueError('too few vals given for list type')

        if len(lfs)<len(vals):
            iext=self._index_extend

            if iext is None:
                raise ValueError('too many vals given for list type')

            dn=len(vals)-len(lfs)
            extfs=[self._default]*dn

            lftfs, rgtfs=lfs[:iext], lfs[iext:]
            lfs=lftfs+extfs+rgtfs

        return [f(s) for f, s in zip(lfs, vals)]

    def __call__(self, v):
        vals=self._vspliter(v)

        return self._convert_list_vals(vals)

#!/usr/bin/env python3

'''
    parser of YAML get
'''

from ..yaml import load_yaml, save_to_yaml

from .util import new_parser, grpdata_pop
from . import util_yaml

# construct parser
def create_parser(parent=None):
    kws=dict(parent=parent, help='YAML unset',
                description='unset key in YAML')
    parser=new_parser('unset', **kws)

    parser.add_argument('file', help='YAML file')
    parser.add_argument('keys', help='key(s) to unset in YAML',
                            nargs='+')

    parser.add_argument('-v', '--verbose', help='verbose of what is done',
                            action='store_true')

    util_yaml.add_key_delim_to_parser(parser)

    # add func
    parser.set_defaults(func=func_of_parser)

    return parser

# function to handle args
def func_of_parser(args):
    data=load_yaml(args.file)

    for s in args.keys:
        if args.verbose:
            print(f'unset `{s}`')

        key=args.keySplit(s)
        grpdata_pop(data, key)

    # save to dest file
    dst=args.file
    if args.verbose:
        print(f'write back to: {dst}')

    save_to_yaml(data, dst, safe_backup=True)

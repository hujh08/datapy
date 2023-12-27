#!/usr/bin/env python3

'''
    parser of YAML get
'''

from ..yaml import load_yaml

from .util import new_parser, grpdata_get
from . import util_yaml

# construct parser
def create_parser(parent=None):
    kws=dict(parent=parent, help='YAML get',
                description='get val from YAML')
    parser=new_parser('get', **kws)

    parser.add_argument('file', help='YAML file')
    parser.add_argument('keys', help='key(s) to get from YAML',
                            nargs='+')

    util_yaml.add_key_delim_to_parser(parser)

    # add func
    parser.set_defaults(func=func_of_parser)

    return parser

# function to handle args
def func_of_parser(args):
    data=load_yaml(args.file)

    for s in args.keys:
        key=args.keySplit(s)
        print(grpdata_get(data, key))

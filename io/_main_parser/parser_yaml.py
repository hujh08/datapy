#!/usr/bin/env python3

'''
    parser for io of YAML
'''

from .util import new_parser, import_local_modules
from . import util_yaml

# import subparser constuctors
modules=['parser_yaml_'+s for s in ['get', 'unset', 'set', 'cp']]
constructors=import_local_modules(modules)

# construct parser
def create_parser(parent=None):
    global constructors

    parser=new_parser('yaml', parent=parent, help='YAML',
                        description='io of YAML')

    # subparsers
    subparsers=parser.add_subparsers(help='io of YAML')

    ## constructors of parsers
    for p in constructors:
        p.create_parser(parent=subparsers)

    return parser

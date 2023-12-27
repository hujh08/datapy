#!/usr/bin/env python3

'''
    parser of YAML copy
'''

import os, shutil

from ..yaml import load_yaml, save_to_yaml

from .util import (new_parser,
                   grpdata_get, grpdata_set, grpdata_contains)
from . import util_yaml

# construct parser
def create_parser(parent=None):
    kws=dict(parent=parent, help='YAML copy',
                description='copy YAML')
    parser=new_parser('cp', **kws)

    parser.add_argument('src', help='source YAML file')
    parser.add_argument('dest', help='dest YAML file or dir')
    parser.add_argument('keys', help=('part of keys to copy, '
                                      'if not given, copy all.'),
                            nargs='*')

    parser.add_argument('-v', '--verbose', help='verbose of what is done',
                            action='store_true')
    parser.add_argument('-f', '--force', help='force to copy if dest exists',
                            action='store_true', dest='cpforce')

    util_yaml.add_key_delim_to_parser(parser)
    parser.add_argument('-r', '--rel-path-keys',
                            help=('list of keys which is relpath to file, '
                                  'if given, change to rel to new file.'),
                            nargs='+', dest='relPathKeys')

    # add func
    parser.set_defaults(func=func_of_parser)

    return parser

# function to handle args
def func_of_parser(args):
    src, dst=args.src, args.dest

    # check
    if not os.path.isfile(src):
        raise FileNotFoundError(f'not a file: {src}')

    if os.path.isdir(dst):
        bname=os.path.basename(src)
        dst=os.path.join(dst, bname)

    if os.path.exists(dst) and not args.cpforce:
        raise FileExistsError(f'dest file exsits: [{dst}]')

    if args.verbose:
        print(f'copy [{src}] ==> [{dst}]')

    # simple copy
    if not (args.keys or args.relPathKeys):
        if args.verbose:
            print('simple copy')

        shutil.copyfile(src, dst)
        return

    # part copy and change of rel-path keys
    data_src=load_yaml(src)

    ## part copy
    if args.keys:
        data_dst={}
        for s in args.keys:
            if args.verbose:
                print(f'copy key: {s}')

            k=args.keySplit(s)
            v=grpdata_get(data_src, k)
            grpdata_set(data_dst, k, v)
    else:
        data_dst=data_src

    ## change rel-path keys to rel new file
    if args.relPathKeys:
        dir_src=os.path.dirname(src)
        dir_dst=os.path.dirname(dst)

        for s in args.relPathKeys:
            k=args.keySplit(s)
            if not grpdata_contains(data_dst, k):
                continue

            if args.verbose:
                print(f'change rel-path key: {s}')

            p=grpdata_get(data_dst, k)
            psrc=os.path.join(dir_src, p)
            pdst=os.path.relpath(psrc, dir_dst)

            if p!=pdst:
                grpdata_set(data_dst, k, pdst)

    # save to dest file
    if args.verbose:
        print(f'save to dest: {dst}')

    save_to_yaml(data_dst, dst)

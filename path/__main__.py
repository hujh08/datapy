#!/usr/bin/env python3

'''
    command line interface to handle path
'''

import os, sys
import argparse

from .tools import find_sub_in_parent

# construct argument parser
parser=argparse.ArgumentParser(description='operations of path')

subparsers=parser.add_subparsers(help='sub-commands')

## abs path
def handle_abspath(args):
    print(os.path.abspath(args.path))

subparser=subparsers.add_parser('abspath',
                                description='output absolute path',
                                help='absolute path')

subparser.add_argument('path', help='path in file system')

subparser.set_defaults(func=handle_abspath)

## relative path
def handle_relpath(args):
    print(os.path.relpath(args.path, args.srcpath))

subparser=subparsers.add_parser('relpath',
                                description='output relative path',
                                help='relative path')

subparser.add_argument('path', help='target path in file system')
subparser.add_argument('srcpath', help='src path for relpath')

subparser.set_defaults(func=handle_relpath)

## expand path by user
def handle_expandpath(args):
    print(os.path.expanduser(args.path))

subparser=subparsers.add_parser('expand',
                                description='expand path for user',
                                help='expand ~ and ~user')

subparser.add_argument('path', help='path in file system')

subparser.set_defaults(func=handle_expandpath)

## join path
def handle_join_path(args):
    print(os.path.join(*args.paths))

subparser=subparsers.add_parser('join',
                                description='output joined paths',
                                help='join path')

subparser.add_argument('paths', help='paths to join',
                        nargs='+')

subparser.set_defaults(func=handle_join_path)

## parent path
def handle_parent(args):
    sub=args.sub
    find_all=args.all
    p=find_sub_in_parent(sub, find_all=find_all,
                            return_rel=(not args.abspath))
    if p is None:
        raise FileNotFoundError(f"'{sub}' not found in parent")

    if not find_all:
        p=[p]

    for t in p:
        print(t)

subparser=subparsers.add_parser('parent',
                                description='find sub in parent',
                                help='sub path in parent')

subparser.add_argument('sub', help='sub path to search in parent')
subparser.add_argument('-b', '--abspath',
                            help='output abs path. default rel path',
                            action='store_true')
subparser.add_argument('-a', '--all',
                            help=('whether to find all paths. '
                                  'default nearest path'),
                            action='store_true')

subparser.set_defaults(func=handle_parent)

# run command
args=parser.parse_args()

args.func(args)

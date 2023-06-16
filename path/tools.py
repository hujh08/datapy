#!/usr/bin/env python3

'''
    simple tools to handle path in file system
'''

import os, sys
from pathlib import Path

__all__=['dirname_posix',
         'find_sub_in_parent', 'add_lib_in_parent']

# posix dirname
def dirname_posix(p):
    '''
        p
    '''
    if p!='/':
        p=p.rstrip('/')

    d=os.path.dirname(p)
    if not d:
        return '.'
    return d

# parent of path
def parent(p):
    return str(path(p).parent)

# parent path containing given sub file
def find_sub_in_parent(sub, return_rel=True, find_all=False):
    '''
        find `sub` in parent path of PWD
            and then join them

        return:
            path of sub if found
            None otherwise

        :param return_rel: bool, default True
            if True, return relative path to PWD
            otherwise, return absolute path

        :param find_all: bool, default True
            if True, return all parent paths containg `sub`
            otherwise return nearest one
    '''
    result=[]

    pwd=os.getcwd()

    visited=set()
    current=pwd
    while current not in visited:
        subcurr=os.path.join(current, sub)

        if os.path.exists(subcurr):
            result.append(subcurr)
            if not find_all:
                break

        visited.add(current)
        current=os.path.dirname(current)

    if not result:
        return None

    if return_rel:
        result=[os.path.relpath(p, pwd) for p in result]

    if not find_all:
        return result[0]

    return result

# add local lib path in parent dir
def add_lib_in_parent(sub):
    '''
        add local lib path in parent
    '''
    libdir=find_sub_in_parent(sub, return_rel=True)
    if libdir not in sys.path:
        sys.path.append(libdir)

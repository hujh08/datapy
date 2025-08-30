#!/usr/bin/env python3

'''
    simple tools to handle path in file system
'''

import os, sys
from pathlib import Path

__all__=['dirname', 'dirname_path',
         'rebase_relpath',
         'find_sub_in_parent', 'add_lib_in_parent']

# posix dirname
def dirname(p):
    '''
        posix style dirname
        e.g.
            dirname('DIR'):
                '.', not ''
            dirname('DIR1/DIR2/'):
                'DIR1', not 'DIR1/DIR2'
    '''
    sep=os.sep
    if p!=sep:
        p=p.rstrip(sep)

    d=os.path.dirname(p)
    if not d:
        return os.curdir
    return d

def dirname_path(p):
    '''
        same as `.dirname`
            implemented by `pathlib.Path`
    '''
    return str(Path(p).parent)

# rebase relpath
def rebase_relpath(path, newbase, oldbase='.'):
    '''
        rebase relpath to new base path
    '''
    path=os.path.join(oldbase, path)
    return os.path.relpath(path, newbase)

# parent path containing given sub file
def find_sub_in_parent(sub, start=None, return_rel=True, find_all=False):
    '''
        find `sub` in parent path of PWD
            and then join them

        return:
            path of sub if found
            None otherwise

        :param start: optional, Path-like
            starting path to search sub

            if not given, use `os.getcwd()`

        :param return_rel: bool, default True
            if True, return relative path to pwd
                not rel to `start`
            otherwise, return absolute path

        :param find_all: bool, default True
            if True, return all parent paths containg `sub`
            otherwise return nearest one
    '''
    result=[]

    if start is None: start=''

    visited=set()
    current=os.path.abspath(start)
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
        result=[os.path.relpath(p) for p in result]

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

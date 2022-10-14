#!/usr/bin/env python3

'''
    simple tools to handle path in file system
'''

import os

__all__=['join_parent_with_sub']

# parent path containing given sub file
def join_parent_with_sub(sub, return_rel=True, find_all=False):
    '''
        find parent path of PWD containing given `sub`
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

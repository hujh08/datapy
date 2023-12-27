#!/usr/bin/env python3

'''
    utitlity for main script
'''

import importlib
import argparse

# new parser or subparser
def new_parser(title=None, parent=None, help=None, **kws):
    '''
        new parser or subparser in parent parser

        :param title, help:
            only needed for subparser
    '''
    if parent is None:
        return argparse.ArgumentParser(**kws)
    return parent.add_parser(title, help=help, **kws)

# import local modules
def import_local_modules(modules):
    # from . import MODULE
    res=[]
    for m in modules:
        res.append(importlib.__import__(m, globals(), level=1))
        # res.append(importlib.import_module('..'+m, __name__))

    return res

# get/set from grouped data
def grpdata_contains(data, key):
    for k in key[:-1]:
        if k not in data:
            return False
        data=data[k]

    return key[-1] in data

def grpdata_get(data, key):
    v=data[key[0]]
    for k in key[1:]:
        v=v[k]

    return v

def grpdata_set(data, key, val):
    for k in key[:-1]:
        if k not in data:
            data[k]={}
        data=data[k]

    data[key[-1]]=val

def grpdata_pop(data, key):
    for k in key[:-1]:
        data=data[k]

    return data.pop(key[-1])
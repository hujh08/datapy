#!/usr/bin/env python3

'''
    auxiliary functions
'''

import os
import numbers
from collections import deque

from PIL import Image

import numpy as np
from scipy.special import comb

# file system
def listdir(path=None, recursive=False, include_dir=False):
    '''
        list files in a directory
            not including special entries '.' and '..'

        Parameters:
            path: str, or None
                dir to list
                if None, use '.' to list
                    but set root dir to ''

            recursive: bool
                whether to subdirectory recursively

            include_dir: bool
                whether to return dir
    '''
    if path is None:
        path='.'
        root=''
    else:
        root=path

    paths=deque([(path, root)])
    while paths:
        path, root=paths.popleft()

        for entry in os.listdir(path):
            pnow=os.path.join(root, entry)

            if not os.path.isdir(pnow):
                yield pnow
                continue

            if include_dir:
                yield pnow

            if recursive:
                paths.append((pnow, pnow))

def read_img(path):
    '''
        read an image,
        
        return an array with shape (w, h, [channal])
                    or other objects, like PIL image objects
            which would be drawn via `ax.imshow`
    '''
    # return Image.imread(path)
    # return np.array(Image.open(path))
    return Image.open(path)

# random
def rand_ind(n):
    '''
        random index for a list with len=`n`
    '''
    return np.random.randint(n)

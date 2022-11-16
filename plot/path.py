#!/usr/bin/env python3

'''
    module to handle path
'''

import numpy as np

import matplotlib.path as mpath

# path codes
def normal_path_codes(codes):
    '''
        normalize given path code

        code could be given in
            str: case-ignored, e.g. 'moveto', 'lineto'
            int: e.g. 1, 2
    '''
    valid_str_codes=['STOP', 'MOVETO', 'LINETO', 'CURVE3', 'CURVE4']
    valid_codes=[getattr(mpath.Path, s) for s in valid_str_codes]
    if isinstance(codes, str):
        code=codes.upper()
        assert code in valid_str_codes, f'invalid code: {codes}'
        return getattr(mpath.Path, code)
    elif isinstance(codes, numbers.Number):
        c=mpath.Path.code_type(codes)
        assert c in valid_codes, f'invalid code: {codes}'

        return c

    # iterable
    return [normal_path_codes(c) for c in codes]

# closed path
class ClosedPath(mpath.Path):
    '''
        different treatment to `closed`
    '''
    def __init__(self, verts, codes=None, closed=False, **kwargs):
        # treatment to `closed`
        self._closed=bool(closed)
        if self._closed:
            verts, codes=self._get_closed_verts_codes(verts, codes)

        super().__init__(verts, codes=codes, **kwargs)

    @staticmethod
    def _get_closed_verts_codes(verts, codes):
        verts=np.asarray(verts)

        if codes is None:
            n=len(verts)
            codes=[mpath.Path.LINETO]*n
            codes[0]=mpath.Path.MOVETO
        elif codes and codes[-1]==mpath.Path.CLOSEPOLY:
            # already closed
            return verts, codes

        verts=np.append(verts, verts[:1], axis=0)
        codes=np.append(codes, mpath.Path.CLOSEPOLY)

        return verts, codes

    # property
    @property
    def vertices(self):
        return super().vertices

    @vertices.setter
    def vertices(self, verts):
        if self._closed:
            verts=np.append(verts, verts[:1], axis=0)

        mpath.Path.vertices.fset(self, verts)

    @property
    def codes(self):
        return super().codes

    @codes.setter
    def codes(self, codes):
        if self._closed:
            codes=np.append(codes, mpath.Path.CLOSEPOLY)

        mpath.Path.codes.fset(self, codes)

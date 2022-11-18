#!/usr/bin/env python3

'''
    module to handle path
'''

import numbers

import numpy as np

import matplotlib.path as mpath
import matplotlib.bezier as mbezier

# path codes
def normal_path_codes(codes):
    '''
        normalize given path code

        code could be given in
            str: case-ignored, e.g. 'moveto', 'lineto'
            int: e.g. 1, 2
    '''
    valid_str_codes=['STOP', 'MOVETO', 'LINETO', 'CURVE3', 'CURVE4', 'CLOSEPOLY']
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

def get_path_code(code):
    '''
        return path code for given str or int
    '''
    assert isinstance(code, str) or isinstance(code, numbers.Number)
    return normal_path_codes(code)

def expand_path_codes(codes, n, ignore_none=True):
    '''
        expand codes to length `n`

        :param codes: None, str, int or list of code
            if list, must have lenght `n`

            if str or int, expand to list of code
                with MOVETO following by series of code

                only support 'LINETO', 'CURVE3', 'CURVE4'
                case-ignored for str

            if None and not `ignore_none`, same as 'LINETO'
    '''
    if codes is None:
        if ignore_none:
            return None

        codes=[mpath.Path.LINETO]*n
        codes[0]=mpath.Path.MOVETO
        return codes

    codes=normal_path_codes(codes)
    if isinstance(codes, numbers.Number):
        ccode=[mpath.Path.LINETO, mpath.Path.CURVE3, mpath.Path.CURVE4]
        if codes not in ccode:
            raise ValueError(f'only support to expand curve code: {ccode}')

        codes=[codes]*n
        codes[0]=mpath.Path.MOVETO

    return codes

def check_path_codes_numverts(codes, nverts, raise_error=False, map_numv=None):
    '''
        check whether codes and number of verts is matched

        more tha 1 control points needed for CURVE3, CURVE4
            if no enough points given, use (0, 0) in matplotlib by default
    '''
    if map_numv is None:
        map_numv=mpath.Path.NUM_VERTICES_FOR_CODE

    if codes is None:
        return True

    if len(codes)!=nverts:
        matched=False
    elif not codes:  # empty codes
        return True
    else:
        if codes[0]!=mpath.Path.MOVETO:
            if raise_error:
                raise ValueError('first code must be MOVETO')
            return False

        matched=True
        codes=np.asarray(codes)

        # split codes segments
        inds_csegs=[0, *(np.nonzero(np.diff(codes))[0]+1), len(codes)]
        for i0, i1 in zip(inds_csegs[:-1], inds_csegs[1:]):
            ci=codes[i0]

            # control points needed
            if ci not in map_numv:
                if raise_error:
                    raise ValueError(f'got unexpected code: {ci}')
                return False

            ni=map_numv[ci]
            if ni<1:
                d={ci: ni}
                raise ValueError(f'invalid map to num of verts for code: {d}')

            if ni==1:
                continue

            if (i1-i0) % ni != 0:
                matched=False
                break

    if not matched and raise_error:
        raise ValueError('mismatch between codes and num of verts')

    return matched

# closed path
class ClosedPath(mpath.Path):
    '''
        class for closed path
            different treatment to `closed`
    '''
    def __init__(self, verts, codes=None, closed=False, strict_codes=False, **kwargs):
        '''
            init of path

            :param codes: None, str, int or list of code
                besides None and codes list, scalar code is supported
                    which would be expanded to list of code

            :param strict_codes: bool, default False
                whether to check whether exists enough control points
        '''
        codes=expand_path_codes(codes, len(verts), ignore_none=True)
        if strict_codes:
            check_path_codes_numverts(codes, len(verts), raise_error=True)

        # treatment to `closed`
        self._closed=bool(closed)
        if self._closed:
            verts, codes=self._get_closed_verts_codes(verts, codes)

        super().__init__(verts, codes=codes, **kwargs)

    def _get_closed_verts_codes(self, verts, codes):
        verts=np.asarray(verts)

        if codes is None:
            codes=self._expand_none_codes(len(verts), closed=False)
        elif codes and codes[-1]==self.CLOSEPOLY:
            # already closed
            return verts, codes

        verts=np.append(verts, verts[:1], axis=0)
        codes=np.append(codes, self.CLOSEPOLY)

        return verts, codes

    def _expand_none_codes(self, n=None, closed=None):
        '''
            expand None codes to series of `LINETO`

            if `closed` is True, last code is `CLOSEPOLY`
        '''
        if closed is None:
            closed=self.closed
        if n is None:
            n=len(self.vertices)

        codes=expand_path_codes(None, n, ignore_none=False)

        if closed:
            codes[-1]=self.CLOSEPOLY

        return codes

    def set_closed(self, closed=True):
        # close the path
        c0=bool(self._closed)
        c1=bool(closed)

        self._closed=c1

        if c0!=c1:  # closed state changes
            verts, codes=self.vertices, self.codes
            if c0:
                assert codes[-1]==self.CLOSEPOLY
                verts=verts[:-1]
                codes=codes[:-1]

            self.vertices=verts
            self.codes=codes

    @property
    def closed(self):
        return bool(self._closed)

    # set vertices and codes
    def set_vertices(self, verts):
        if self._closed:
            verts=np.append(verts, verts[:1], axis=0)

        self._set_parent_prop('vertices', verts)

    def set_codes(self, codes):
        if self._closed:
            if codes is None:
                codes=self._expand_none_codes()
            else:
                codes=np.append(codes, self.CLOSEPOLY)

        self._set_parent_prop('codes', codes)

    ## auxiliary
    def _set_parent_prop(self, prop, val):
        '''
            set property in parent class

            parent is relative to this class, not class of `self`
                that is first base class
        '''
        p=getattr(__class__.__bases__[0], prop)
        p.fset(self, val)

    ## property
    @property
    def vertices(self):
        return super().vertices

    @vertices.setter
    def vertices(self, verts):
        self.set_vertices(verts)

    @property
    def codes(self):
        return super().codes

    @codes.setter
    def codes(self, codes):
        self.set_codes(codes)

# path for Bezier curve, beyond CURVE4
class BezierPath(ClosedPath):
    def __init__(self, verts, ts=100, closed=False, **kwargs):
        '''
            init of Bezier curve

            :param ts: int or array like
                interpolating point along the curve
        '''
        self._bseg=mbezier.BezierSegment(verts)

        # points to interpolate along curve
        self._set_ts(ts)

        # use vertices from interpolation
        verts=self._bseg(self._ts)
        super().__init__(verts, codes=None, closed=closed, **kwargs)

        ## allow update underlying path
        self._allow_update_base_path=False

    # control points
    def get_control_points(self):
        return self._bseg.control_points

    @property
    def control_points(self):
        return self.get_control_points()

    # underlying path
    def get_base_path(self):
        '''
            underlying path
                which uses vertices from interpolation
        '''
        return mpath.Path(self.vertices, self.codes)

    def _permiss_base_path_update(self):
        class AllowUpdate:
            def __init__(self, p):
                self._path=p

            def __enter__(self):
                self._path._allow_update_base_path=True
                return self._path

            def __exit__(self, *args):
                self._path._allow_update_base_path=False

        return AllowUpdate(self)

    def _update_base_path(self):
        '''
            update underlying path
        '''
        with self._permiss_base_path_update():
            self.vertices=self._bseg(self._ts)
            self.codes=None

    def set_closed(self, closed=True):
        with self._permiss_base_path_update():
            super().set_closed(closed)

    ## modify vertices and codes of underlying path
    def set_vertices(self, verts):
        if not self._allow_update_base_path:
            raise AttributeError('cannot set vertices of Bezier curve directly')
        super().set_vertices(verts)

    def set_codes(self, codes):
        if not self._allow_update_base_path:
            raise AttributeError('cannot set codes of Bezier curve directly')
        super().set_codes(codes)

    # handle `ts`
    def _set_ts(self, ts):
        if isinstance(ts, numbers.Number):
            ts=np.linspace(0, 1, ts)
        else:
            ts=np.asarray(ts)
            assert ts.ndim==1

        self._ts=ts

    def set_ts(self, ts):
        self._set_ts(ts)
        self._update_base_path()

    # str representation
    def __repr__(self):
        sname=type(self).__name__
        cpoints=self.get_control_points()
        return f'{sname}({repr(cpoints)})'

# segments from structured list of points
def parse_path_segs(*segs):
    '''
        parse path segments specified by structured list of points
        return
            vertices, codes

        each segment is represented by list of collection of control points
            None in the end for closed path
    '''
    verts=[]
    codes=[]
    for seg in segs:
        vis, cis=parse_path_seg(seg)
        verts.extend(vis)
        codes.extend(cis)

    return verts, codes

## auxiliary functions
def parse_path_seg(seg):
    '''
        parse one path segment

        each segment is represented by list of collection of control points
            None in the end for closed path
    '''
    seg=list(seg)

    closed=False
    if seg and seg[-1] is None:
        seg=seg[:-1]
        closed=True  # None in end for closed

    if not seg:
        raise ValueError('empty path seg')

    verts=[]
    codes=[]

    # first point
    p0=seg.pop(0)

    if not is_type_of_path_point(p0):
        cpoints=list(p0)  # control points
        p0=cpoints.pop(0)

        if cpoints:  # drop empty list
            seg.insert(0, cpoints)

    verts.append(p0)
    codes.append(mpath.Path.MOVETO)

    # control points
    map_ncp_code={1: mpath.Path.LINETO, # num of control points: code
                  2: mpath.Path.CURVE3, 
                  3: mpath.Path.CURVE4}
    for cpoints in seg:
        if is_type_of_path_point(cpoints):
            # scalar point, as lineto
            verts.append(cpoints)
            codes.append(mpath.Path.LINETO)
            continue

        n=len(cpoints)
        if n not in map_ncp_code:
            raise ValueError(f'unexpected num of control points: {n}')

        verts.extend(cpoints)
        codes.extend([map_ncp_code[n]]*n)

    if not all(map(is_type_of_path_point, verts)):
        raise TypeError('meet wrong point type in seg')

    if closed:
        verts.append(verts[0])
        codes.append(mpath.Path.CLOSEPOLY)

    return verts, codes

def is_type_of_path_point(p):
    return len(p)==2 and \
           all([isinstance(t, numbers.Number) for t in p])
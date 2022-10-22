#!/usr/bin/env python3

'''
    frequently used parameters
'''

import matplotlib.pyplot as plt

from .size import convert_unit

__all__=['set_rc_for_tex', 'set_rc_for_tex_cls',
         'set_rc_for_mnras']

# params for `Rect.create_axes`
def params_create_axes(style):
    '''
        frequently used args for `Rect.create_axes`

        current support:
            'tight grid': used for axes in tight grid
                tick with direction=in,
                          tick in 4 edges
                nlnx, nbny: no xlabel if not left
                            no ylabel if not bottom
    '''
    if style=='tight grid':
        return dict(tick_params=dict(which='both', direction='in', top=True, right=True),
                    nlnx=True, nbny=True)

    raise ValueError('unexpected style: \'%s\'' % style)

# set rcParams for figure in Tex
def set_rc_for_tex(width, fontsize, height=None, unit='pt_tex'):
    r'''
        set rcParams for figure in Tex
            mainly figsize and fontsize

        Figure is ofter inserted in pdf
            with width equal to `textwidth` (two column)
                             or `clumnwidth` (one column)
        and the fontsize should be same as that of caption

        These value could be got by putting `\the` commands in Tex document:
        just as following
            `\the\textwidth`
            `\the\columnwidth`
            `\the\fontdimen6\font`

        :param height: None or float
            often textheight
            if None, use `width`*2

        use `plt.rcdefaults()` to restore default plt.rcParams
    '''
    if height is None:
        height=width*2

    # figsize
    u=convert_unit(unit, dest='inch')
    w_inch=u*width
    h_inch=u*height

    plt.rc('figure', figsize=(w_inch, h_inch))

    # fontsize
    u=convert_unit(unit, dest='points')
    plt.rc('font', size=fontsize*u)

## set for Tex documentclass
_rcset_tex_cls={}
def _register_rcset_texcls(name):
    '''
        register a function to set rc for Tex cls
    '''
    def wrapper(func):
        _rcset_tex_cls[name]=func
        return func
    return wrapper

def set_rc_for_tex_cls(texcls, *args, **kwargs):
    '''
        set rc for a Tex classes
    '''
    if texcls not in _rcset_tex_cls:
        raise ValueError(f'not support for Tex class yet: {texcls}')

    func=_rcset_tex_cls[texcls]
    return func(*args, **kwargs)

@_register_rcset_texcls('mnras')
def set_rc_for_mnras(twocol=False, scale=1):
    r'''
        set for `mnras` document class

        values of
            `textwidth`, `columnwidth`, `fontsize`
        are got by following `\the` command
            `\the\textwidth`
            `\the\columnwidth`
            `\the\fontdimen6\font`

        :param twocol: bool, default False
            insert figure in two columns or one

            if True, use `scale*textwidth`
            otherwise, use `scale*columnwidth`
    '''
    textwidth, columnwidth=508, 244  # in unit Tex pt
    textheight=682
    fontsize=8.5  # fontsize of figure caption

    width=textwidth if twocol else columnwidth

    set_rc_for_tex(width=width*scale, fontsize=fontsize,
                   height=textheight, unit='pt_tex')

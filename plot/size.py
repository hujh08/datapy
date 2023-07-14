#!/usr/bin/env python3

'''
    Unit of size used in figure

    frequently used: inch, points (pt), mm
        - An inch is 25.4 mm.
        - For TeX, 1 pt is 1/72.27 in, which is 0.351459804 mm.
        - For most other softwares, 1 pt is 1/72 in, which is 0.352777778 mm.
          Also called Postscript Point, in TeX this is called a big point (bp)
'''

import numbers

import matplotlib.pyplot as plt
from matplotlib import font_manager

# Unit to inch
Units=dict(
    inches=1.,      # 25.4 mm
    pt_tex=1/72.27, # For TeX, 1 pt is 1/72.27 inches
    points=1/72,    # In typography, a point is 1/72 inches. 
                    #    In TeX this is called a big point (bp)
    cm=1/2.54,
    )
Units['mm']=0.1*Units['cm']

## some alias
Units['pts']=Units['points']
Units['inch']=Units['inches']

# function

## convert between units
def convert_unit(src, dest='inch'):
    '''
        convert src unit to another ('inch' by default)
    '''
    d=Units[src]
    if dest[:4]!='inch':
        d=d/Units[dest]

    return d

## fontsize
def fontsize_in_pts(size=None):
    '''
        convert fontsize to value in unit points

        Parameters:
            size: float, None, or str
                font size

                if float,
                    absolute value of fontsize in unit points

                if str,
                    'xx-small', 'x-small', 'small', 'medium', 'large', 
                    'x-large', 'xx-large', 'larger', 'smaller'
                see `matplotlib.font_manager.font_scalings`
                    for details
    '''
    if isinstance(size, numbers.Number):
        return size

    fontsize=plt.rcParams['font.size']
    if size is None:
        return fontsize

    assert isinstance(size, str), \
        'only support float, str or None for fontsize, ' \
        'but got %s' % (type(size).__name__)

    font_scalings=font_manager.font_scalings
    assert size in font_scalings, \
        'only allow fontsize in %s, ' \
        'but got \'%s\'' % (str(list(font_scalings.keys())),
                                size)
    s=font_scalings[size]
    return s*fontsize
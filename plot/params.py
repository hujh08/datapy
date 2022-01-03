#!/usr/bin/env python3

'''
    frequently used parameters
'''

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
        return dict(tick_params=dict(direction='in', top=True, right=True),
                    nlnx=True, nbny=True)

    raise ValueError('unexpected style: \'%s\'' % style)

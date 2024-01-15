#!/usr/bin/env python3

'''
    utilities for plot
'''

from ._tools_layout import check_axis

__all__=['set_axis_share', 'copy_line_to_ax']

# axis share
def set_axis_share(axis, ax0, ax1, *axs):
    '''
        set axes to share with other

        Parameters:
            axis: 'x' or 'y'
                axis to share
    '''
    check_axis(axis)

    # grouper for axis share: `matplotlib.cbook.Grouper`
    grp=getattr(ax0, 'get_shared_%s_axes' % axis)()

    # share axis 
    axs=(ax1,)+axs
    set_share_with=getattr(ax0, 'share'+axis)
    for axi in axs:
        if grp.joined(ax0, axi): # already shared
            continue
        set_share_with(axi)

# copy line
def copy_line_to_ax(ax, line, **kws):
    '''
        copy line to a new ax
    '''
    xs, ys=line.get_xydata().T

    # set default props
    for k in ['color', 'alpha', 'lw', 'ls']:
        p=getattr(line, f'get_{k}')()
        kws.setdefault(k, p)

    return ax.plot(xs, ys, **kws)

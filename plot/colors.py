#!/usr/bin/env python3

'''
    functions to handle color in matplotlib
'''

# color cycle

def get_next_color_in_cycle(ax):
    '''
        get next color in color cycle of an axes
    '''
    ltmp,=ax.plot([])
    color=ltmp.get_color()
    ltmp.remove()

    return color

# color map

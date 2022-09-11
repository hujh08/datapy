#!/usr/bin/env python3

'''
    functions to handle legend tasks in matplotlib
'''

from matplotlib.legend import Legend

import matplotlib.collections as mcoll
import matplotlib.patches as mpatches
import matplotlib.legend_handler as mhandler

# update default handler map
def update_default_handler_map(map_handlers):
    Legend.update_default_handler_map(map_handlers)

# legend handlers

## for contour plot
##     matplotlib 3.5.3 seems to fail to create proper legend for contour
class ContourHandler(mhandler.HandlerBase):
    def __init__(self, fill, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if fill:
            self._handler=mhandler.HandlerPatch()
            self._create_patch=self._create_patch_fill
        else:
            self._handler=mhandler.HandlerLineCollection()
            self._create_patch=self._create_patch_nonfill

    # translate to another patch
    def _create_patch_fill(self, orig_handle):
        patch=mpatches.Rectangle((0, 0), 1, 1,
                    facecolor=orig_handle.get_facecolor()[0],
                    hatch=orig_handle.get_hatch(),
                    alpha=orig_handle.get_alpha())
        return patch

    def _create_patch_nonfill(self, orig_handle):
        patch=mcoll.LineCollection(None)
        patch.update_from(orig_handle)
        return patch

    # create artists
    def create_artists(self, legend, orig_handle, *args):
        print('create_artists in ContourHandler')
        
        patch=self._create_patch(orig_handle)
        return self._handler.create_artists(legend, patch, *args)

def update_handler_for_contour(cset, fill=False):
    # update handler map for a contour plot
    handler=handler_fill if fill else handler_nonfill
    update_default_handler_map({cset: handler})

handler_nonfill=ContourHandler(fill=False)
handler_fill=ContourHandler(fill=True)

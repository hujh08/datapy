#!/usr/bin/env python3

'''
    functions to handle legend tasks in matplotlib
'''

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import matplotlib.legend as mlegend

import matplotlib.collections as mcoll
import matplotlib.patches as mpatches
import matplotlib.legend_handler as mhandler

from .size import convert_unit, fontsize_in_pts

# update default handler map
def update_default_handler_map(map_handlers):
    mlegend.Legend.update_default_handler_map(map_handlers)

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
        patch=self._create_patch(orig_handle)
        return self._handler.create_artists(legend, patch, *args)

def update_handler_for_contour(cset, fill=False):
    # update handler map for a contour plot
    handler=handler_fill if fill else handler_nonfill
    update_default_handler_map({cset: handler})

handler_nonfill=ContourHandler(fill=False)
handler_fill=ContourHandler(fill=True)

# additional flexible methods to add legend

## wrapper of ax.legend
def add_legend(ax, *args,
                    onlylabel=False, labelrha=False,
                    alpha=None,
                    labelalpha=None, handlealpha=None,
                    labels_as_key=True,
                    **kwargs):
    '''
        wrapper of `ax.legend` with new features:
            1, frameon
                default is changed to False
            2, labelcolor
                support more type:
                    new str:
                        'facecolor', 'fc',
                        'edgecolor', 'ec',
                    callable:
                        with signature
                            f(handle) ==> color
            3, new args for alpha of texts and handles:
                alpha, labelalpha, handlealpha: None or float
                    if some not None,
                        force corresponding alpha to given value

                if `alpha` is not None, and one of others two args None,
                    change it to value of arg `alpha`

            4, onlylabel: bool
                only draw text label,
                    no marker or line in legend is hidden

                label is distinguished by color
                    use facecolor of handle by default

                new param: labelrha: bool, default False
                    horizontal alignment of label, 'left' for 'right'
                        True for right

                    in `ax.legend`,
                        it is controlled by param `markerfirst`
            5, labels_as_key: bool, default True
                whether to use labels as key for handles
                    works when only `labels` given

                if True:
                    handle is matched by label-handle map,
                        constructed from `ax.get_legend_handles_labels`:
                            axhs, axls=ax.get_legend_handles_labels()
                            map_axlh=dict(zip(axls, axhs))
                            handles=[map_axhl[l] for l in labels]
                if False:
                    matched by order as `ax.legend`,
                        which means
                            axhs, _=ax.get_legend_handles_labels()
                            handles, labels=zip(*zip(axhs, labels))
    '''
    kwargs.setdefault('frameon', False)

    # parse of handles, labels
    *args, kwargs=\
        _parse_legend_from_args(ax, *args, **kwargs,
                                    labels_as_key=labels_as_key)

    # alpha
    if alpha is not None:
        if labelalpha is None:
            labelalpha=alpha
        if handlealpha is None:
            handlealpha=alpha

    # labelcolor
    new_color_getter=None  # new setter of color
    if 'labelcolor' in kwargs:
        lc=kwargs['labelcolor']
        if isinstance(lc, str):
            if lc in ['facecolor', 'fc', 'edgecolor', 'ec']:
                fc=(lc in ['facecolor', 'fc'])
                new_color_getter=_color_getter_from_legend_handle(fc=fc)
                kwargs.pop('labelcolor')
        elif callable(lc):
            new_color_getter=kwargs.pop('labelcolor')

    # onlylabel mode
    if onlylabel:
        # zero length of handle
        kwargs.setdefault('handlelength', 0)
        kwargs.setdefault('handletextpad', 0)

        # horizontal alignment
        kwargs.setdefault('markerfirst', not labelrha)

        # color: use facecolor by default
        if new_color_getter is None:
            if kwargs.get('labelcolor', None) is None:
                new_color_getter=_color_getter_from_legend_handle()
                if labelalpha is None:
                    labelalpha=1

    # draw legend
    pleg=ax.legend(*args, **kwargs)

    # new color/alpha for Text
    texts=pleg.get_texts()
    handles=pleg.legendHandles
    if new_color_getter is not None:
        for t, h in zip(texts, handles):
            t.set_color(new_color_getter(h))

    if labelalpha is not None:
        for t, h in zip(texts, handles):
            t.set_alpha(labelalpha)

    # new alpha for handle
    if onlylabel:
        for h in handles:
            h.set_visible(False)   # not supported to remove for legend

        return pleg

    if handlealpha is not None:
        for h in handles:
            h.set_alpha(handlealpha)

    return pleg

## legend with only colored text
def add_color_texts(ax, texts, colors,
                        loc=(0.1, 0.1),
                        ha='left', va='bottom',
                        rowspacing=None,
                        **kwargs):
    '''
        add colored texts

        Parameters:
            texts, colors: list of text/color
                text and color to add

                same length

            loc: 2-tuple of float, default (0.1, 0.1)
                positon of collection of Texts
                    in axes coordinate

                anchor of the value is given by `ha` and `va`

            ha: 'left', 'center', 'right', default 'left'
            va: 'bottom', 'center', 'top', default 'bottom'
                horizontal/vertival alignment

                these two params control both followings:
                    anchor of Text collection for param `loc`
                    ha/va of each Text object

            rowspacing: None, float, default None
                spacing between rows of texts
                    in unit fontsize
                        given by param `fontsize` or rc:font.size

                if None,
                    use :rc:`legend.labelspacing`

            optional kwargs: passed to `ax.text`
    '''
    texts=list(texts)
    colors=list(colors)

    if len(texts)!=len(colors):
        s='mismatch length between `texts` and `colors`'
        raise ValueError(s)

    # check ha/va
    s_ha=['left', 'center', 'right']
    assert ha in s_ha, f'only allow `ha` in {s_ha}'

    s_va=['bottom', 'center', 'top']
    assert va in s_va, f'only allow `va` in {s_va}'

    # rowspacing to unit inches
    if rowspacing is None:
        rowspacing=plt.rcParams['legend.labelspacing']

    fontsize=fontsize_in_pts(kwargs.get('fontsize', None))
    t_inch=convert_unit('points', 'inch')  # inch/points
    
    tdy_inch=(1+rowspacing)*fontsize*t_inch  # spacing between baseline of text

    offy_inch=0   # offset along y axis in unit inches
    numt=len(texts)
    if va=='bottom':
        offy_inch+=tdy_inch*(numt-1)
    elif va=='center':
        offy_inch+=tdy_inch*(numt-1)/2

    # draw text
    transInches=ax.figure.dpi_scale_trans
    transAxes=ax.transAxes

    tx, ty=loc

    for t, c in zip(texts, colors):
        # offset transform
        offset=mtrans.ScaledTranslation(0, offy_inch, transInches)

        ax.text(tx, ty, t, color=c, **kwargs,
                    transform=transAxes+offset,
                    ha=ha, va=va)
        offy_inch-=tdy_inch

def add_legend_color_text(ax, *args, colors=None,
                               labels_as_key=True, **kwargs):
    '''
        add legend with only colored text labels

        Parameters:
            colors: None, str, iterable, or callable
                color or color getter (from handle)

                if None:
                    use 'fc'
                if str:
                    'facecolor', 'fc'
                    'edgecolor', 'ec'
                if iterable:
                    list of colors
                if callable:
                    f(handle) ==> color

            labels_as_key: bool, default True
                whether to use labels as key for handles
                    when only `labels` given
                
                if not,
                    match by order, same as `ax.legend`

    '''
    handles, labels, kwargs=\
        _parse_legend_from_args(ax, *args, **kwargs,
                                    labels_as_key=labels_as_key)
    kwargs.pop('handler_map', None)  # no need handler_map after

    # color getter
    color_getter=None
    if colors is None:
        color_getter=_color_getter_from_legend_handle()
    elif isinstance(colors, str):
        s_colors=['facecolor', 'fc', 'edgecolor', 'ec']
        if colors not in s_colors:
            raise ValueError(f'only allowed str `colors` is {s_colors}, '
                             f'but got {colors}')

        fc=(lc in ['facecolor', 'fc'])
        color_getter=_color_getter_from_legend_handle(fc=fc)
    elif callable(colors):
        color_getter=colors
    else:
        colors=list(colors)

    if color_getter is not None:
        colors=list(map(color_getter, handles))

    # draw text
    return add_color_texts(ax, labels, colors, **kwargs)

## auxiliary functions
def _parse_legend_from_args(ax, *args, labels_as_key=True, **kwargs):
    '''
        parse legend handles and labels from args

        Signatures of legend (same as `ax.legend`):
            legend()
            legend(labels)
            legend(handles, labels)
            legend(labels=labels)
            legend(handles=handles)
            legend(handles=handles, labels=labels)

        Changes:
            1, if only labels given and `labels_as_key` True,
                handle is matched by label-handle map,
                    constructed from `ax.get_legend_handles_labels`:
                        axhs, axls=ax.get_legend_handles_labels()
                        map_axlh=dict(zip(axls, axhs))
                        handles=[map_axhl[l] for l in labels]
                instead of by order in `ax.legend`,
                    which means
                        axhs, _=ax.get_legend_handles_labels()
                        handles, labels=zip(*zip(axhs, labels))
            2, if both handles and lables given, but with mismatch length
                raise ValueError

        Returns:
            handles, labels, kwargs
    '''
    if ('handles' in kwargs or 'labels' in kwargs) and args:
         raise ValueError('cannot mix positonal and keyword arguments '
                          'for `handles` and `labels`')

    handler_map=kwargs.get('handler_map', None)

    # parse args, kwargs
    if args:
        if len(args)==1:
            handles, labels=None, args[0]
        else:
            handles, labels=args
    else:
        handles=kwargs.pop('handles', None)
        labels=kwargs.pop('labels', None)

    # handles and labels
    if handles is not None and labels is not None:
        if len(handles)!=len(labels):
            s='mismatch length between `handles` and `labels`'
            raise ValueError(s)

        return handles, labels, kwargs

    ## handles None or labels None
    if labels is None and handles is not None:
        labels=[h.get_label() for h in handles]
        return handles, labels, kwargs

    axhs, axls=ax.get_legend_handles_labels(legend_handler_map=handler_map)
    if handles is None and labels is None:
        return axhs, axls, kwargs

    ### handles None, labels not None
    if labels_as_key:  # match by key
        map_axhl=dict(zip(axls, axhs))
        handles=[map_axhl[l] for l in labels]
    else:  # match by order
        handles=axhs[:len(labels)]

    return handles, labels, kwargs

def _color_getter_from_legend_handle(fc=True):
    '''
        color getter from legend handle

        :param fc: bool
            whether to get facecolor or edgecolor
    '''
    if fc:
        basef=lambda h: h.get_facecolor()
    else:
        basef=lambda h: h.get_edgecolor()

    def cgetter(handle):
        color=basef(handle)
        if np.ndim(color)==2:  # for PathCollection
            color=color[0]

        return color

    return cgetter

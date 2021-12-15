#!/usr/bin/env python3

'''
    image marker

    Given a list of image, show and then mark it one by one
'''

import os
import numbers
from collections import OrderedDict
import warnings

import pickle

from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt

from ._funcs import listdir, rand_ind, read_img

class imgMarker:
    '''
        class for image marker
    '''
    def __init__(self, imgs=None, iter_rand=None, mark_forward=None, **kwargs):
        '''
            initiation work

            Parameters:
                imgs: str, list of str/tuple, or None
                    specify list of images to view and mark
                    list of pairs (name, file path), each for an image
                        would be constructed, implicitly

                    if None, construct an empty list

                    if str, path for a dir or a file
                        which is a directory storing image files, or
                                 a catalog file

                    if list, elements should be tuple or str
                        which is (name, path) or just path

                iter_rand: bool, or None (for default)
                    whether to iter images randomly

                mark_forward: bool, or None
                    whether to go forward straightly after mark

                optional kwargs: other args for GUI
        '''
        self._init_imgs_holder()
        self._init_gui()

        if imgs is None:
            return

        # str type for `imgs`: file or dir
        if isinstance(imgs, str):
            if not os.path.exists(imgs):
                raise Exception('Error: no such file or dir: [%s]'
                                    % imgs)

            if os.path.isdir(imgs):
                self.load_img_dir(imgs)
            else:
                self.load_img_catalog(imgs)

        # GUI args
        if iter_rand is not None:
            self._set_gui_iter_rand(iter_rand)

        if mark_forward is not None:
            self._set_gui_mark_forward(mark_forward)

        if kwargs:
            self._set_gui_args(**kwargs)

    # basic methods of image holder
    def _init_imgs_holder(self):
        '''
            init setup of images' holder
        '''
        self._names_imgs=[]    # list of image name
        self._map_imgfile={}  # map name to file

    def _add_image(self, name, fname):
        '''
            basic method to add image

            (name, fname): name and file of image
        '''
        if name in self._map_imgfile:
            warnings.warn('warning: to modify existed image, [%s]'
                                % name)
        else:
            self._names_imgs.append(name)

        self._map_imgfile[name]=fname

    def _sort_imgs(self, key=None, reverse=False):
        '''
            sort image list
            
            Parameters:
                key: None, callable, list-like
                    same as `list.sort`, except list-like type

                    if list-like given, it means weights of element weights

                reverse: bool
                    ascending (default) or descending
        '''
        if key is not None and not callable(key):
            # list-like
            assert hasattr(key, '__getitem__')
            n=len(self._names_imgs)
            indsort=sorted(range(n), key=lambda i: key[i])
            self._names_imgs[:]=[self._names_imgs[i] for i in indsort]
        else:
            self._names_imgs.sort(key=key, reverse=reverse)

    def _get_imgpath_by_name(self, name):
        '''
            get image file by its name
        '''
        return self._map_imgfile[name]

    def _get_imgname_by_ind(self, i):
        '''
            get image name by index
        '''
        return self._names_imgs[i]

    def _get_imgpath_by_ind(self, i):
        '''
            get image path by index
        '''
        name=self._get_imgname_by_ind(i)
        return self._get_imgpath_by_name(name)

    def _imgname_contained(self, name):
        '''
            whether an image name is contained in the list
        '''
        return name in self._map_imgfile

    @property
    def _num_imgs(self):
        # number of images
        return len(self._names_imgs)

    def _iter_imgnames(self, index_iter=None):
        '''
            only iter along image names

            Parameters:
                index_iter: iter
                    iter for image index

                    if None, use sequential iter by default
        '''
        if index_iter is None:
            index_iter=range(len(self._names_imgs))

        for i in index_iter:
            yield self._get_imgname_by_ind(i)

    def _iter_imgs(self, index_iter=None):
        '''
            iter of images

            return tuple (name, path)
        '''
        for name in self._iter_imgnames(index_iter):
            fname=self._get_imgpath_by_name(name)

            yield name, fname

    # image list
    ## constructor
    def add_image(self, fname, name=None):
        '''
            add an image file to the list

            Parameters:
                fname: path
                    path for the image to add

                name: str or None
                    name of the image

                    if None, use `basename` of the path
                        with also suffix striped
        '''
        assert os.path.isfile(fname)

        if name is None:
            name=os.path.basename(fname)

            # remove suffix
            t=name.rsplit('.', maxsplit=1)[0]
            if t!='':
                name=t

        self._add_image(name, fname)

    def load_img_list(self, imglist):
        '''
            load list of images

            basic loader.
                other loaders would finally go to it
        '''
        for entry in imglist:
            if isinstance(entry, str):
                self.add_image(entry)
            else:  # otherwist (name, fname)
                name, fname=entry
                self.add_image(fname, name)

    def load_img_dir(self, dir_imgs, recursive=False, sort=True):
        '''
            load images from a directory

            Parameters:
                dir_imgs: str
                    image dir

                recursive: bool
                    if true, also list files recursively in subdirs

                sort: bool
                    if true, sort image list after being loaded
        '''
        assert os.path.isdir(dir_imgs)

        imgs=list(listdir(dir_imgs, recursive=recursive, include_dir=False))

        self.load_img_list(imgs)

        if sort:
            self._sort_imgs()

    def load_img_catalog(self, fname, skiprows=None, skipcomment=True,
                                      strict=False):
        '''
            load images in catalog file

            Parameters:
                fname: str
                    path of catalog

                skiprows: list-like or int
                    line number to skip

                    0-indexed
                    first line is 0

                skipcomment: bool
                    skip comment line, led by '#'

                strict: bool
                    whether to treat line in catalog strictly
                        only first 1 or 2 fields are needed

                    if True, raise Exception if not the case
        '''
        assert os.path.isfile(fname)

        if isinstance(skiprows, numbers.Integral):
            skiprows=[skiprows]

        imgs=[]
        with open(fname) as f:
            for i, line in enumerate(f):
                if skipcomment and line[0]=='#':
                    continue

                if skiprows is not None and i in skiprows:
                    continue

                fields=line.split()
                if strict and len(fields) not in [1, 2]:
                    msg='Error: abnormal line met: line number %i' \
                            % (i+1)
                    raise Exception(msg)

                if not strict and len(fields)==0:
                    # skip blank line
                    continue

                if len(fields)==1:
                    imgs.append(fields[0])
                else:
                    imgs.append(fields[:2])

        self.load_img_list(imgs)

    ## getter
    @property
    def imgs(self):
        '''
            return list of tuple (name, path)
        '''
        return list(self._iter_imgs())

    ## setter
    def sort_imgs(self, **kwargs):
        '''
            sort image list to a specified order

            order of image might be used in iter

            `kwargs`: same as `list.sort`,
                except `key` could be list-like, giving weights of elements
        '''
        self._sort_imgs(**kwargs)

    # GUI: default setup
    def _init_gui(self):
        '''
            init gui

            2 things would be initiated:
                - arguments to control GUI performance
                - some neccessary objects
                    like callbacks setup, global record buffer
        '''
        print('init gui args')
        self._args_gui={}

        # graphical interface
        self._init_gui_graph()

        # images iterator
        self._init_gui_args_imgiter()

        # record system
        self._init_gui_record()

        # other setup for GUI
        self._alloc_new_gui_args(
            # mark and then forward directly
            mark_forward=False,
            )

    def _alloc_new_gui_args(self, **kwargs):
        '''
            allocate new GUI args
        '''
        # avoid confict between arguments
        assert all([k not in self._args_gui for k in kwargs])

        self._args_gui.update(kwargs)

    def _get_gui_arg(self, k):
        # get gui arg
        return self._args_gui[k]

    def _set_gui_args(self, **kwargs):
        '''
            set gui args
        '''
        # only set existed args
        assert all([k in self._args_gui for k in kwargs])

        self._args_gui.update(kwargs)

    ## mark forward
    def _is_gui_mark_forward(self):
        '''
            whether marker goes forward immediately after mark
        '''
        return self._args_gui['mark_forward']

    def _set_gui_mark_forward(self, mod=True):
        '''
            set `mark_forward` state
        '''
        self._args_gui['mark_forward']=bool(mod)

    # GUI: graphical interface
    def _init_gui_graph(self):
        '''
            init graphical interface,
                including arguments for plot, and
                          callbacks for events
        '''
        print('init args for graph')

        # layout of plt axes
        self._init_gui_args_graph_plt()

        # key press event
        self._init_gui_graph_keypress()

    def _new_gui_graph(self):
        '''
            new graphical interface
        '''
        print('new graph interface for GUI')

        self._new_gui_graph_plt()
        self._new_gui_graph_callbacks_bind()

    def _clear_gui_graph(self):
        '''
            clean work for GUI graphical interface
            It would be done when quit
        '''
        print('clear graph of GUI')
        self._clear_gui_graph_plt()

    ## plot axes
    def _init_gui_args_graph_plt(self):
        '''
            init args for plot in GUI

            2 axes would be added,
                for image drawing and text displaying respectively
        '''
        print('init args for graph plot')

        self._alloc_new_gui_args(
            # figure size
            figsize=None,

            # ratio of ax size for text to image
            ratio_txt_img=0.2,

            # width and height of 2 axes as a whole, with respect to fig
            h_axs_fig=0.9,
            w_axs_fig=0.9,

            # ratio of left margin to right
            ratio_margin_lr=0.7,

            # text
            xy_text=(0.3, 0.95), # in ax.transAxes
            ha_text='left',
            va_text='top'
            )

    def _new_gui_graph_plt(self):
        '''
            new plot axes for GUI
        '''
        print('new graph plt')
        # figure
        figsize=self._get_gui_arg('figsize')
        self._fig=plt.figure(figsize=figsize)

        w, h=self._fig.get_size_inches()

        # setup for 2 axes
        r_ti=self._get_gui_arg('ratio_txt_img')

        h_axs_fig=self._get_gui_arg('h_axs_fig')
        w_axs_fig=self._get_gui_arg('w_axs_fig')

        r_lr=self._get_gui_arg('ratio_margin_lr')

        ## equal size for image ax
        w_img_inch=w*w_axs_fig/(1+r_ti)
        h_img_inch=h*h_axs_fig

        if h_img_inch<w_img_inch:
            w_axs_fig=(h_img_inch/w)*(1+r_ti)
        else:
            h_axs_fig=w_img_inch/h

        ## left, bottom
        assert 0<w_axs_fig<=1 and 0<h_axs_fig<=1
        laxs=(1-w_axs_fig)*r_lr/(1+r_lr)
        baxs=(1-h_axs_fig)/2

        # image axes
        left=laxs
        width=w_axs_fig/(1+r_ti)

        btm=baxs
        height=h_axs_fig

        rect=[left, btm, width, height]
        self._imgax=self._fig.add_axes(rect, label='image')
        self._imgax.axis('off')

        # text box
        left+=width
        width*=r_ti

        rect=[left, btm, width, height]
        self._txtax=self._fig.add_axes(rect, label='text')
        self._txtax.axis('off')

    def _clear_gui_graph_plt(self):
        '''
            clear work for GUI plot
        '''
        print('clear gui plot')
        del self._fig
        del self._imgax
        del self._txtax

    ## callbacks responding to event
    def _new_gui_graph_callbacks_bind(self):
        '''
            bind new callbacks for GUI
        '''
        print('new callbacks bind')

        # disconnect default binding of key press
        canvas=self._fig.canvas
        if canvas.manager is not None:
            canvas.mpl_disconnect(canvas.manager.key_press_handler_id)

        canvas.mpl_connect('close_event', self._close)
        canvas.mpl_connect('key_press_event', self._on_press_key)

    def _close(self, event):
        '''
            callback for close event
        '''
        # print('close event:', event.__dict__)
        print('close event:', list(event.__dict__.keys()))

        self._clear_gui()

        print()

    def _on_press_key(self, event):
        '''
            callback for key press event
        '''
        # print('key press event:', event.__dict__)
        print('key press event:', list(event.__dict__.keys()))

        iax=event.inaxes
        print('    inaxes:', iax if iax is None else iax._label)

        key=event.key
        print('    key: [%s]' % key)

        f=self._get_callback_by_key(key)
        if f is not None:
            f()

        print()

    ## key press event
    def _init_gui_graph_keypress(self):
        '''
            init setup for key press event

            two types of key press events:
                special action:
                    press for some special actions
                        e.g. quit, forward [or backward (to support later)]

                mark action:
                    press to mark image
        '''
        print('init args for graph key press')

        # 'callbacks' for keys. also specify valid actions
        #     further wrap for real callbacks, by `_get_callback_by_key`
        self._callbacks_act=dict(quit=self._gui_quit,
                                 forward=self._gui_forward,
                                 mark=self._gui_img_mark)

        # special action keys
        self._keys_act=dict(q='quit',
                            right='forward' # direction key
                           )

        # mark keys: remember adding order, to be used in marks selecting
        self._keys_mark=OrderedDict()

    def _get_callback_by_key(self, key):
        '''
            get real callback for pressed key

            Parameters:
                key: str
                    pressed key name
        '''
        if key in self._keys_act:
            a=self._keys_act[key]
            return self._callbacks_act[a]
        elif key in self._keys_mark:
            f=self._callbacks_act['mark']
            m=self._keys_mark[key]
            return lambda: f(m)

        return None  # None for non-existed key

    ### setter for key press
    def _modify_key_act(self, act, key):
        '''
            change action key to a new one

            it should only be called before GUI start
        '''
        if act not in self._callbacks_act or act=='mark':
            # invalid act
            raise Exception('Error: invalid act: [%s]' % act)
            return

        if key in self._keys_act:
            a0=self._keys_act[key]
            if act!=a0:
                raise Exception('Error: key [%s] already exists for [%s], '
                                'cannot change to [%s]' % (key, a0, act))
            return

        map_act={a: k for k, a in self._keys_act.items()
                    if a!='mark'}
        oldkey=map_act[act]

        self._keys_act[key]=act
        self._keys_act.pop(oldkey)

    #### keys for mark
    _MARK_UNDECIDED='UNDECIDED'
    def _add_key_mark(self, key, mark, force=False):
        '''
            add a mark key
        '''
        if key in self._keys_act:
            a0=self._keys_act[key]
            raise Exception('Error: key [%s] already exists for [%s], '
                            'cannot change to mark [%s]' % (key, a0, mark))

        if mark==self._MARK_UNDECIDED:
            raise Exception('Error: mark [%s] already used for '
                            'special mark of undecided' % (mark,))


        if key in self._keys_mark:
            m0=self._keys_mark[key]
            if m0!=mark and not force:
                raise Exception('Error: key [%s] already exists to mark [%s], '
                                'cannot change to [%s]' % (key, m0, mark))
            elif m0==mark:
                return

        self._keys_mark[key]=mark

    def _clear_keys_mark(self):
        '''
            clear setup of mark keys
        '''
        self._keys_mark.clear()

    def _is_undecided_mark(self, mark):
        # whether the mark is the 'Undecided' mark
        return mark==self._MARK_UNDECIDED

    ### getter for markers
    def _get_marks_registered(self, priority=None):
        '''
            return all registered marks
                with some order

            Parameters:
                priority: None, dict, or list (of marks)
                    priorities of the mark,
                        larger value for higher priority
                    
                    allow to give only part of marks

                    if None, use the order in OrderedDict

                    for marks not in `priorities`
                        assume they have priorities lower than those specified
                            and then use initial order in OrderedDict
        '''
        marks=[]
        for k in self._keys_mark.values():
            if k not in marks:
                marks.append(k)

        if priority is not None:
            if not isinstance(priority, dict):
                # larger value for higher priority
                priority={k: -i for i, k in enumerate(priority)}

            marks_nop=[]
            marks_p=[]
            for i, m in enumerate(marks):
                if m not in priority:
                    marks_nop.append(m)
                    continue

                p=priority[m]
                marks_p.append((m, (p, -i)))

            marks_p.sort(key=lambda t: t[1], reverse=True)

            marks=[t[0] for t in marks_p]+marks_nop

        return marks

    ### base callbacks
    def _gui_quit(self):
        # quit GUI
        print('quit')
        plt.close(self._fig)

    def _gui_forward(self):
        # forward: mv to next image
        print('forward')

        self._imgiter_next()

    def _gui_img_mark(self, mark):
        '''
            mark an image

            additional arg needed, not final callback
            It would be wrapped via `_get_callback_by_key`
        '''
        # mark an image
        print('mark image')
        print('    mark:', mark)

        self._add_record_mark(mark)

        if self._is_gui_mark_forward():
            self._gui_forward()

    ## updater of graphical interface
    def _update_gui_graph(self):
        # update GUI graph
        print('update graph')

        self._update_gui_graph_plt()

    def _update_gui_graph_plt(self):
        # update plot
        print('update graph plot')

        self._draw_image()
        self._disp_img_info()

        self._fig.canvas.draw()

    def _draw_image(self):
        '''
            draw current image
                refered by `self._ind_now`
        '''
        print('draw image')
        imgpath=self._get_path_imgnow()
        print('    img ind:', self._ind_now+1)
        print('    img path:',  imgpath)

        img=read_img(imgpath)

        ax=self._imgax
        if ax.images:
            ax.images.clear()

        ax.imshow(img)

    def _disp_img_info(self):
        '''
            display info of current image
        '''
        print('display img info')

        imgname=self._get_name_imgnow()
        imgpath=self._get_path_imgnow()
        print('    name:', imgname)
        print('    path:', imgpath)

        # message to display
        msgs=[]

        i=self._ind_now+1
        n=self._num_imgs
        msgs.append('%i/%i' % (i, n))
        msgs.append(imgname)

        msgs.append('')
        msgs.append(self._get_txt_brief_record())

        msg='\n'.join(msgs)
        print('    lines of msg:', len(msgs))

        # display in text ax
        xy=self._get_gui_arg('xy_text')
        ha=self._get_gui_arg('ha_text')
        va=self._get_gui_arg('va_text')
        ax=self._txtax
        if ax.texts:
            ax.texts.clear()
        ax.text(*xy, msg, transform=ax.transAxes, ha=ha, va=va)

    # GUI: images iterator
    def _init_gui_args_imgiter(self):
        '''
            init args of images iterator
        '''
        print('init args for imgiter')

        self._alloc_new_gui_args(
            # iter along images randomly
            iter_rand=True,
            )

    def _is_gui_iter_rand(self):
        '''
            whether to iter randomly
        '''
        return self._args_gui['iter_rand']

    def _set_gui_iter_rand(self, mod=True):
        '''
            set to iter randomly
        '''
        self._args_gui['iter_rand']=bool(mod)

    def _new_gui_imgiter(self):
        '''
            new image iter for GUI
        '''
        print('new image iter')

        # rand or seq
        self._ind_now=0

        if self._is_gui_iter_rand():
            print('    rand iter')
            self._get_next_ind=self._next_ind_by_rand

            self._ind_now=self._get_next_ind()
        else:
            print('    seq iter')
            self._get_next_ind=self._next_ind_by_seq

        print('    init ind:', self._ind_now)

    def _clear_gui_imgiter(self):
        '''
            clear work for image iter
        '''
        print('clear image iter')

        del self._ind_now
        del self._get_next_ind

    ## getter for current image
    def _get_path_imgnow(self):
        # return path of current image
        return self._get_imgpath_by_ind(self._ind_now)

    def _get_name_imgnow(self):
        # return name of current image
        return self._get_imgname_by_ind(self._ind_now)

    ## next method for image iter
    def _imgiter_next(self):
        '''
            move to next image
        '''
        print('iter to next')
        print('    ind now:', self._ind_now)

        ind_next=self._get_next_ind()
        if ind_next is None:
            print('    already last:', self._ind_now)
            return

        if ind_next==self._ind_now:
            print('    same image returned')
            return

        # merge record
        self._merge_staged_record()

        # change `_ind_now`
        print('change ind_now')
        self._ind_now=ind_next
        print('    move to:', self._ind_now)

        # update
        self._update_gui()

    ## calculate next index
    def _next_ind_by_seq(self):
        '''
            return next index in sequence

            if exceeding max length, return None
        '''
        if self._ind_now>=self._num_imgs-1:
            return None
        else:
            return self._ind_now+1

    def _next_ind_by_rand(self):
        '''
            return next index in rand
        '''
        return rand_ind(self._num_imgs)

    # GUI: record system
    def _init_gui_record(self):
        '''
            init of record system
            
            buffer for global record is allocated
        '''
        print('init gui record')
        self._record_global={}    # global

    def _new_gui_record(self):
        '''
            new record system for GUI

            two parts: local and global
                global one is allocated at `_init_gui_record`
                local one is allocated here
        '''
        print('new gui record')
        self._record_staged={}    # local record

    def _clear_gui_record(self):
        '''
            clear work for record system

            just merge staged record to global
        '''
        print('clear gui record')

        self._merge_staged_record()

        assert not self._record_staged  # empty staged record
        del self._record_staged

    ## local record
    def _add_record_mark(self, mark):
        '''
            add mark to local record
        '''
        print('add mark to local record:', mark)
        print('    old record:', self._record_staged)

        self._record_staged['mark']=mark
        print('    new record:', self._record_staged)

    def _to_record_entry_from_stage(self):
        '''
            convert staged record
                to an entry in globa record
        '''
        if not self._record_staged or \
           'mark' not in self._record_staged:
            return {}

        # maybe more record entries to support later
        m0=self._record_staged['mark']
        return {'mark': {m0: 1}}

    ## global record
    def _merge_record_for_img(self, name, record, is_reckey=False):
        '''
            merge record of an image

            :param is_reckey: bool
                if True, the arg `name` is global record key
                otherwise, image name
        '''
        if not record:  # empty record
            return

        if 'mark' not in record: # no mark
            return
        mrk=record['mark']

        if not is_reckey:
            reckey=self._get_reckey_by_imgname(name)
        else:
            reckey=name

        if reckey not in self._record_global:
            self._record_global[reckey]={}
        imgrec=self._record_global[reckey]

        if 'mark' not in imgrec:
            imgrec['mark']={}
        mrkrec=imgrec['mark']

        for m0, cnt in mrk.items():
            if m0 not in mrkrec:
                mrkrec[m0]=0
            mrkrec[m0]+=cnt

    def _merge_staged_record(self):
        '''
            merge local record to global

            only called before `_ind_now` changes or GUI close
        '''
        print('merge local record')
        print('    ind now:', self._ind_now)
        print('    staged record:', self._record_staged)
        print('    all recs:', len(self._record_global))

        record=self._to_record_entry_from_stage()
        if not record:
            print('    emtpy staged')
            return

        reckey=self._get_reckey_imgnow()
        self._merge_record_for_img(reckey, record, is_reckey=True)

        print('    new rec for img:', self._record_global[reckey])
        print('    new recs:', len(self._record_global))

        # clear local record
        self._record_staged.clear()

    ## construct of record key
    def _get_reckey_by_imgname(self, name):
        '''
            return record key for an image

            basic constructor of record key
        '''
        return name

    def _get_imgname_from_reckey(self, reckey):
        '''
            re-parse image name from record key
        '''
        return reckey

    def _get_reckey_by_ind(self, ind):
        '''
            return record key for an image index
        '''
        name=self._get_imgname_by_ind(ind)
        return self._get_reckey_by_imgname(name)

    def _get_reckey_imgnow(self):
        '''
            get a record key for current image
        '''
        return self._get_reckey_by_ind(self._ind_now)

    ## getter
    def _get_rec_by_imgname(self, name):
        '''
            return record for an image
        '''
        reckey=self._get_reckey_by_imgname(name)
        if reckey not in self._record_global:
            return None
        return self._record_global[reckey]

    def _get_marks_by_imgname(self, name):
        '''
            return record of marks for an image
        '''
        rec=self._get_rec_by_imgname(name)
        if rec is None:
            return None
        return rec['mark']

    ## text for record: used for display
    def _get_txt_brief_record(self, indent=''):
        '''
            get text of brief for record
                which is then used in GUI display
        '''
        lines=[]

        # statistic of marks
        ntot_img=self._num_imgs
        nmrk_img=0

        cnts_mark={}
        for key, rec in self._record_global.items():
            imgname=self._get_imgname_from_reckey(key)
            if not self._imgname_contained(imgname):
                continue

            nmrk_img+=1
            mrk=rec['mark']
            for k, n in mrk.items():
                if k not in cnts_mark:
                    cnts_mark[k]=0
                cnts_mark[k]+=n

        lines.append('marked: %i/%i' % (nmrk_img, ntot_img))

        ntotmrk=sum(cnts_mark.values())
        lines.append('counts: %i' % (ntotmrk,))

        if cnts_mark:  # count for each mark
            marks=sorted(cnts_mark.keys())
            cnts=[cnts_mark[m] for m in marks]

            # string for each mark: first letter+count
            smrks=[m[0]+str(c) for m,c in zip(marks, cnts)]

            lines.append('+'.join(smrks))

        return '\n'.join([indent+t for t in lines])

    # GUI: running
    def _new_gui(self):
        '''
            prepare work for a new GUI
        '''
        print('new GUI')

        self._new_gui_graph()
        self._new_gui_imgiter()
        self._new_gui_record()

    def _update_gui(self):
        '''
            update GUI state
        '''
        print('update GUI')

        self._update_gui_graph()

    def _clear_gui(self):
        '''
            clear work for GUI
        '''
        print('clear GUI')

        self._clear_gui_record()
        self._clear_gui_imgiter()
        self._clear_gui_graph()

    def _mainloop(self):
        '''
            mainloop for the GUI
        '''
        print('mainloop of GUI')
        print()

        plt.show()

    # GUI: user methods
    def run(self):
        '''
            run the marker
        '''
        print('run marker')

        self._new_gui()
        self._update_gui()
        self._mainloop()

    ## setter of GUI args
    def set_mark_forward(self, *args, **kwargs):
        '''
            set marker to go forward immediately after mark
        '''
        self._set_gui_mark_forward(*args, **kwargs)

    def set_iter_rand(self, *args, **kwargs):
        '''
            set marker to iter along images randomly
        '''
        self._set_gui_iter_rand(*args, **kwargs)

    def set_gui_args(self, **kwargs):
        '''
            general set for GUI args
        '''
        self._set_gui_args(**kwargs)

    ## key press
    @property
    def marks(self):
        '''
            all marks
        '''
        return self._get_marks_registered()

    def get_marks_registered(self, priority=None):
        '''
            return all registered marks
                with some order

            Parameters:
                priority: None, dict, or list (of marks)
                    priorities of the mark
                        allow to give only part of marks
        '''
        return self._get_marks_registered(priority=priority)

    ### setter of key press
    def modify_key_act(self, act, key):
        '''
            modify an action key
                like quit, forward, et. al.
        '''
        self._modify_key_act(act, key)

    def add_key_mark(self, key, mark, force=False):
        '''
            add a key for mark
        '''
        self._add_key_mark(key, mark, force=force)

    def add_keys_mark_by_dict(self, d, force=False):
        '''
            add mark keys from a dict
        '''
        for k, m in d.items():
            self.add_key_mark(k, m, force=force)

    def set_keys_mark_to(self, d):
        '''
            set mark keys with a dict
        '''
        self.clear_keys_mark()
        self.add_keys_mark_by_dict(d)

    def clear_keys_mark(self):
        '''
            clear setup of mark keys
        '''
        self._clear_keys_mark()

    ## GUI record
    @property
    def record(self):
        '''
            return record from GUI
        '''
        return self._record_global

    def brief_record(self, **kwargs):
        '''
            print string for brief of record
        '''
        print(self._get_txt_brief_record(**kwargs))

    def get_marks_by_name(self, name):
        '''
            return marks for an image name
        '''
        return self._get_marks_by_imgname(name)

    def get_marks_images(self, prior_marks=None, sigma=2,
                            stat_check=False,
                            return_dict=False):
        '''
            return marks for all images

            Some images may have more than one marks done
            Assume there are M different marks
                   and totaly N times of marks is done
            Strategy to pick out one is as following:
                1, choose marks with max count

                2, if more than 1 have max count,
                    leave image mark as special mark 'UNDECIDED'
                    or choose one based on the priority

                3, otherwise, only 1 has max count
                    use Bayesian strategy to decide if it is significant
                        that means other counts <= mincount
                            mincount = mean + t*std
                        mean and std could be infered from Bayesian posterior
                            with binomial likelihood and uniform prior
                        Here, simply, mean=p=k/n,
                                      std=sqrt(n*p*(1-p))/n

                    if not significant, then mark as 'UNDECIDED'

                4, special case, counts for other marks are 0
                    simple criterion would not work
                    But current posterior is simple P(p)=(n+1) * p^n

                    Then the probability for p less than 1/M
                                        (same probabiltiy for each mark)
                        is P(p<1/M)=1/M^n
                    if it's greater than a significance,
                        then the result is considered as undecided
                    just like situation 2),
                        leave image mark as special mark 'UNDECIDED'
                        or choose one based on the priority

                    It is also probability of the result in N i.i.d tests
                        M results for each test, (same probability)
                    if outcome of all results is (n1, n2, ..),
                        ni for ith result, and n1+n2+.N
                    then its probability is
                        (n1! n2! .../N!) * 1/M^N
                    for ni=N, and other nj=0, it's 1/M^N

            Parameters:
                prior_marks: None, dict, list-like
                    priority of marks

                sigma: float
                    sigma for statistic calculation

                stat_check: bool
                    whether to do statistical significance check

                    if True, check is done
                        and use special mark 'UNDECIDED' for not significant result
                    
                    otherwise just pick mark with highest priority

                return_dict: bool
                    whether to return dict

                    if False, return list of tuples (name, mark)
        '''
        # marks ordered by priority
        marksp=self._get_marks_registered(priority=prior_marks)
        nmarks=len(marksp)
        assert nmarks>=1

        # significance from sigma
        if stat_check:
            assert sigma>0
            alpha=(1-erf(sigma/np.sqrt(2)))/2  # one-tail

            ## min n for 1/M^n > signficance
            mincnt=-np.log(alpha)/np.log(len(marksp))
            assert mincnt>0

        # loop to pick mark for images
        marks_imgs=[]
        for name in self._iter_imgnames():
            marks_rec=self._get_marks_by_imgname(name)
            if marks_rec is None:
                continue

            cnts=np.array([marks_rec.get(m, 0) for m in marksp])
            assert np.all(cnts>=0)

            ntot=np.sum(cnts)
            if ntot<=0:
                continue

            # choose marks with max count
            mc=np.max(cnts)
            inds=[i for i, c in enumerate(cnts) if c==mc]
            mrk0=marksp[inds[0]]

            if not stat_check:
                marks_imgs.append((name, mrk0))
                continue

            if len(inds)>1:   # more than 1 max count
                mrk0=self._MARK_UNDECIDED
                marks_imgs.append((name, mrk0))
                continue

            if mc==ntot:  # only one mark
                if mc<mincnt:
                    mrk0=self._MARK_UNDECIDED
                marks_imgs.append((name, mrk0))
                continue

            d=np.sqrt(mc*(ntot-mc))  # sqrt(n*p*(1-p)), p=k/n
            lb=mc-sigma*d

            cnts[inds[0]]=-1
            mc1=np.max(cnts)  # 2nd max
            if mc1>lb:
                mrk0=self._MARK_UNDECIDED
            marks_imgs.append((name, mrk0))

        if return_dict:
            return dict(marks_imgs)

        return marks_imgs

    def write_marks_to_file(self, fname, update=True, **kwargs):
        '''
            write marks to a file

            Parameters:
                fname: path
                    path of file to write

                update: bool
                    whether to update mark file if it exists

                    if True, output order would be that of image name

                kwargs: optional arguments
                    used in `get_marks_images`

                    not include `return_dict`
        '''
        marks=self.get_marks_images(**kwargs, return_dict=False)

        if update and os.path.isfile(fname):
            marks=dict(marks)
            with open(fname) as f:
                for line in f:
                    fields=line.split()
                    if len(fields)!=2:
                        continue

                    name, mark=fields
                    if name not in marks:
                        marks[name]=mark

            marks=[(k, marks[k]) for k in sorted(marks.keys())]

        # write
        with open(fname, 'w') as f:
            for name, mark in marks:
                f.write('%s %s\n' % (name, mark))

        return len(marks)

    ## dump and restore record
    def dump_record(self, fname):
        '''
            dump global record

            Parameters:
                fname: str
                    path of file to dump record
        '''
        with open(fname, 'wb') as f:
            pickle.dump(self._record_global, f)

    def restore_record(self, fname, merge=True, ignore_nonexists=True):
        '''
            restore previous record

            Parameters:
                fname: str
                    path of file to restore

                merge: bool
                    whether to merge dumped record to current

                    if False, just replace current record with dumped

                ignore_nonexists: bool
                    whether to ignore non-existed file given by `fname`

                    if False, an Exception would be raised if not exists
        '''
        if not os.path.exists(fname):
            if not ignore_nonexists:
                raise Exception('Error: file not exists, [%s]' % fname)
            else:
                print('file not exists, [%s]' % fname)
                return
                
        with open(fname, 'rb') as f:
            records=pickle.load(f)

        if not merge:
            self._record_global.clear()
            self._record_global.update(records)
            return

        for key, rec in records.items():
            self._merge_record_for_img(key, rec, is_reckey=True)

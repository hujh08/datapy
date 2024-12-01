#!/usr/bin/env python3

'''
    io of HDF5 file

    data in HDF5 file is represented by dict in routine
'''

import h5py

import pandas as pd

__all__=['load_hdf5', #'load_hdf5_attrs',
         'save_to_hdf5',
         'load_hdf5_pds', 'save_to_hdf5_pds',]

# load hdf5
def load_hdf5(path_or_obj, fields=None, key_attrs=None, ignore_dset_attrs=True,
                           dict_for_list_fields=None,
                           ignore_nonexists=False):
    '''
        load data in HDF5

        --------
        Data
            in Group is represented by dict
            in Dataset by array
                if attrs returned for Dataset, use tuple: (data, attrs) or (attrs,)

        keyword argument `fields` could be given for partly loading
        It has slightly different meaning for Group and Dataset
            for Group, it's for sub-group/dataset, as HDF5
            for dataset, it's for items in dataset, as np.ndarray
                e.g. slice, list of indices, or tuple of indices list

        -----
        attrs is metadata attached to both Group and Dataset
            attrs would be ignored for any loading of Dataset and partly loading of Group,
                except attrs query is specified explicitly
            complete loading of Group would contain attrs
                which would load all data, naturally including attrs

        They could be considered as special sub-group in parent Group and Dataset
            For attrs of Group, just same way as its normal sub-Group and sub-Dataset

        A special `fields` format is designed for attrs of Dataset
            To query attrs, 3 way is allowed ('attrs' could be changed by `key_attrs`):
                - raw string 'attrs':
                    return all attrs if existed

                - dict {'attrs': attrs-fields}:
                    return {'attrs': part-of-attrs-by-fields}

                - 2-tuple (item-like, dict):
                    last element for attrs query
                    return 2-tuple (data-field, attrs-field)

        ignore all attrs query only set `key_attrs` to None or False

        --------
        Parameters:
            path_or_obj: filename, h5py.File, or h5py.Group
                HDF5 file or object to load

                data in this object specified by `fields` would be returned
                    if Dataset type, no fields should be given

            fields: None (or bool True), field name, item-like, list of names, nested dict or list
                which data in HDF5 would be returned

                How this keyword is used depends firstly on type of `path_or_obj`

                different types:
                    - None or bool True: return all data in `path_or_obj`
                        if Group, return dict
                        otherwise return array (ignore attrs)
                        NOTE: None, True: totally same meaning
                            but True has more direct meaning, True to get all data

                    - field name: str or bytes
                        In this case, `path_or_obj` must be h5py.File or h5py.Group
                        specify a Group or Dataset in `path_or_obj`

                    - item-like: slice, list of indices, or tuple (of slice or list)
                        In this case, `path_or_obj` must be h5py.Dataset
                        used as same way as getitem of `np.ndarray`

                    - list of names: collection of `field names`
                        In this case, `path_or_obj` must be h5py.File or h5py.Group
                        specify sub-group or datset to return
                            data returned has type list
                                                or dict (with those names for keys)
                                specified by `dict_for_list_fields`

                    - nested dict: dict with value for sub-fields
                        In this case, `path_or_obj` must be h5py.File or h5py.Group
                            its key is to specify sub-group/dataset
                            its value is of previous types: None (or bool), field name, tuple or list
                                       or other nested
                                For bool type value:
                                    if a key has value None or bool True, get all data
                                    if the value is bool False, ignore this key

                    - nested list: list of sub-fields
                        Similar as nested dict,
                            `path_or_obj` must be h5py.File or h5py.Group
                        always used to construct nested data structure
                            fields given by elements are relative to current group

                only allow None (or bool True) and list of names for attrs query

                Examples for dataset query:
                    None         # all data, including attrs if not `ignore_dset_attrs`
                    ()           # empty tuple, all data, but no attrs
                    (slice(None, 100), [1, 2])      # part of data

                    'attrs'                         # only attrs
                    {'attrs': ['attr1', 'attr2']}   # part of attrs

                    ([1, 2, ..], {'attrs': [...]})  # both data and attrs

            key_attrs: None, bool, or str, default None
                whether to fetch attrs from HDF5 object
                    also inherited by sub-group/dataset

                if `key_attrs` is None or bool False, ignore all attrs

                if True and attrs existed
                    attrs returned as an item in dict with key `key_attrs`

                different types for `key_attrs`:
                    if None or bool False, ignore attrs

                    if bool True,
                        use default key 'attrs' for attrs from HDF5
                            in output dict

                    if str, that is key in output dict for attrs

            ignore_dset_attrs: bool, default True
                whether to ignore attrs of dataset
                    except explicit attrs query in passed `fields`

                It works only when None or True passed to `fields`
                    meaning loading all data

            dict_for_list_fields: None, or bool, default bool True
                whether to return dict for list of names given in `fields`,
                    also inherited by sub-fields in case for nested structure

                works only when
                    load data from Group and
                    list type given in `fields`
                force to True in other cases

                if None,
                    True for names for all elements in list
                    otherwise False (in case for nested list)

                if False, return list of data, each element for a field name

                if True, return dict with key
                    except when nested list given
                        exception raised in this case

            ignore_nonexists: bool, default False
                whether to ignore nonexists key

                if not ignore,
                    Exception raised if non-exists
                otherwise,
                    nonexists key will skip
                        except when `dict_for_list_fields`
                            fill by None for the non-exists
    '''
    # key attrs
    if isinstance(key_attrs, bool):
        key_attrs='attrs' if key_attrs else None

    kws=dict(key_attrs=key_attrs,
             dict_for_list_fields=dict_for_list_fields,
             ignore_dset_attrs=ignore_dset_attrs,
             ignore_nonexists=ignore_nonexists)

    # filename
    if isinstance(path_or_obj, (str, bytes)):
        with h5py.File(path_or_obj, 'r') as h5f:
            return _get_data_from_group(h5f, fields=fields, **kws)

    # h5py object
    assert isinstance(path_or_obj, (h5py.File, h5py.Group))

    return _get_data_from_group(path_or_obj, fields=fields, **kws)

def load_hdf5_attrs(path_or_obj, fields):
    '''
        only load attrs from HDF5 file or instance
    '''
    raise NotImplementedError('to implement later')

## real work place

### get data from Dataset
def _get_data_from_dataset(dset, fields=None, key_attrs=None, ignore_dset_attrs=True):
    '''
        get data from h5py.Dataset
            return tuple if attrs got
            otherwise ndarray

        `key_attrs` would be ignored if arg `fields` is not for attrs query

        :param fields: None, slice, list of int, tuple of indices list, or dict (or special 2-tuple)
            dict is used to get attrs, see `_get_attrs_field_of_hdf5` for detail
                only support key by `key_attrs`

            special tuple with len==2 and second is dict
                also used to query attrs and data together
            the dict element is same as previous with only key `key_attrs`
            and if first element is bool False, ignore data

        :param ignore_dset_attrs: bool, default True
            whether to ignore attrs of dataset
                except explicit attrs query in passed `fields`
    '''
    # attrs query
    if isinstance(fields, str) or isinstance(fields, dict) or \
       (isinstance(fields, tuple) and \
        len(fields)==2 and isinstance(fields[1], dict)):

        if key_attrs is None:
            raise ValueError('got an attrs query for Dataset. '
                             'But `key_attrs` is None.')

        # raw string for just attrs
        if isinstance(fields, str):
            if fields!=key_attrs:
                raise ValueError(f'only allow `key_attrs` "{key_attrs}" as str `fields`, '
                                 f'but got "{fields}".')

            return (_get_attrs_field_of_hdf5(dset.attrs),)

        # 2-tuple and dict
        if isinstance(fields, tuple):
            kd, ka=fields  # 2-tuple
        else:
            ka=fields      # dict
            kd=False

        # check dict fields for attrs
        keys=list(ka.keys())
        if len(keys)!=1 or keys[0]!=key_attrs:
            raise ValueError( 'wrong format for dict `fields` of Dataset. '
                             f'only allow `key_attrs` "{key_attrs}" as key')

        k=key_attrs
        a=_get_attrs_field_of_hdf5(dset.attrs, fields[k])

        if isinstance(kd, bool) and not kd:
            return (a,)

        # query data
        d=_get_data_from_dataset_only_data(dset, kd)

        return (d, a)

    # only query data
    res=_get_data_from_dataset_only_data(dset, fields)

    ## attrs
    if (not ignore_dset_attrs) and \
       (key_attrs is not None) and \
       (fields is None or (isinstance(fields, bool) and fields)):

        if dset.attrs:
            a=_get_attrs_field_of_hdf5(dset.attrs)
            res=(res, a)

    return res

def _get_data_from_dataset_only_data(dset, fields=None):
    '''
        only get data from dataset, not including attrs

        :param fields: None, slice, list of int, or tuple of indices list
    '''
    if fields is None or (isinstance(fields, bool) and fields):  # True or None to load all
        return dset[()]
    return dset.__getitem__(fields)

### get data from Group
def _get_data_from_group(grp, fields=None, dict_for_list_fields=None,
                            ignore_nonexists=False, **kwargs):
    '''
        get data from h5py.Group

        work is recursive

        attrs of root group by arg `grp` would be ignored
            except `fields` is given by None (or bool True)
                which will return all data in group, including attrs
    '''
    # all data in group
    if fields is None or (isinstance(fields, bool) and fields):  # True or None to load all
        return _get_data_from_group_total(grp, **kwargs)

    # one field
    if isinstance(fields, (str, bytes)):
        return _get_data_from_group_by_name(grp, fields, **kwargs,
                                    ignore_nonexists=ignore_nonexists)

    # nested structure
    kws=dict(**kwargs, dict_for_list_fields=dict_for_list_fields,
                ignore_nonexists=ignore_nonexists)

    ## dict for `fields`
    if isinstance(fields, dict):
        return _get_data_from_group_by_dict(grp, fields, **kws)

    ## list for `fields`
    fields=list(fields)
    return _get_data_from_group_by_list(grp, fields, **kws)

#### get from Group by different types for `fields`
def _get_data_from_group_total(grp, key_attrs=None, **kwargs):
    '''
        get data from Group in total
        return a dict
    '''
    res={}

    for k in grp.keys():
        d=grp[k]

        if isinstance(d, h5py.Group):
            res[k]=_get_data_from_group_total(d, key_attrs=key_attrs, **kwargs)
            continue

        res[k]=_get_data_from_dataset(d, key_attrs=key_attrs, **kwargs)

    # attrs
    if key_attrs is not None and grp.attrs:
        if key_attrs in res:
            s=(f'name exists as `key_attrs` "{key_attrs}" '
               f'in group of HDF5 file: {grp.file}')
            raise ValueError(s)

        res[key_attrs]=dict(grp.attrs)

    return res

def _get_data_from_group_by_name(grp, name, key_attrs=None,
                                        ignore_nonexists=False, **kwargs):
    '''
        get sub-Group/Dataset with name `name`
    '''
    obj=_get_sub_or_attrs_of_group(grp, name, key_attrs=key_attrs,
                                        ignore_nonexists=ignore_nonexists)
    if obj is None:
        return None

    # attrs
    if isinstance(obj, h5py.AttributeManager):
        return _get_attrs_field_of_hdf5(obj)

    # sub-Dataset or -Group
    if isinstance(obj, h5py.Dataset):
        return _get_data_from_dataset(obj, key_attrs=key_attrs, **kwargs)

    return _get_data_from_group_total(obj, key_attrs=key_attrs, **kwargs)

def _get_data_from_group_by_dict(grp, fields, key_attrs=None,
                                      dict_for_list_fields=None,
                                      ignore_nonexists=False, **kwargs):
    '''
        get data by dict given for `fields`
            key: name of sub-dataset/group in root `grp`
            value: fields for the key, any type valid in `_get_data_from_group`
                if None or bool True, get all data
                if bool False, ignore it

        :param dict_for_list_fields: None, bool
            passed to sub-fields with type `list`
    '''
    res={}
    for k, v in fields.items():
        # bool type for dict value
        if isinstance(v, bool) and not v: # bool False, ignore this key
                continue

        d=_get_sub_or_attrs_of_group(grp, k, key_attrs=key_attrs,
                                        ignore_nonexists=ignore_nonexists)
        if d is None:
            continue

        # attrs
        if isinstance(d, h5py.AttributeManager):
            res[k]=_get_attrs_field_of_hdf5(d, v)
            continue

        # sub-group or -dataset
        if isinstance(d, h5py.Dataset):
            res[k]=_get_data_from_dataset(d, v, key_attrs=key_attrs, **kwargs)
            continue

        d1=_get_data_from_group(d, v, **kwargs, key_attrs=key_attrs,
                                    ignore_nonexists=ignore_nonexists,
                                    dict_for_list_fields=dict_for_list_fields)
        if d1 is None:
            continue
        res[k]=d1

    return res

def _get_data_from_group_by_list(grp, fields, dict_for_list_fields=None, **kwargs):
    '''
        get data by list given for `fields`

        :param key_attrs: None or str
            key to store attrs in dict
                passed to sub-field

        :param dict_for_list_fields: None, or bool
            args given to `dict_for_list_fields` in other funcs
                are passed to this func finally

            also inherited by nested list

            if None,
                True for names for all elements in list
                otherwise False (in case for nested list)

            if False, return list of data, each element for a field name

            if True, return dict with key
    '''
    if dict_for_list_fields is None:
        if all([isinstance(k, (str, bytes)) for k in fields]):
            dict_for_list_fields=True
        else:
            dict_for_list_fields=False

    kws=dict(**kwargs, dict_for_list_fields=dict_for_list_fields)

    # get data by list recursively
    fields_load=[]
    res=[]
    for k in fields:
        d=_get_data_from_group(grp, k, **kws)
        if d is None and dict_for_list_fields:
            continue
        res.append(d)
        fields_load.append(k)

    # True for `dict_for_list_fields`
    if dict_for_list_fields:
        res=dict(zip(fields_load, res))

    return res

#### auxiliary funcs of group
def _get_sub_or_attrs_of_group(grp, name, key_attrs=None, ignore_nonexists=False):
    '''
        get sub-group/dataset or dataset for a group
    '''
    if name not in grp:
        if key_attrs is not None and name==key_attrs:  # only load attrs
            return grp.attrs

        if not ignore_nonexists:
            s=f'name "{name}" not exists in group of HDF5 file: {grp.file}'
            raise ValueError(s)
        else:
            return None

    return grp[name]

def _is_list_type_fields(fields):
    '''
        list type for arg `fields`
    '''
    return hasattr(fields, '__iter__') and \
           not isinstance(fields, (str, bytes, dict))

### get attrs
def _get_attrs_field_of_hdf5(attrs, fields=None):
    '''
        get fields of HDF5 attrs

        :param fields: None, bool True, or list
    '''
    if fields is None or (isinstance(fields, bool) and fields):  # True or None to load all
        return dict(attrs)

    return {k: attrs[k] for k in fields}

# save data to hdf5
def save_to_hdf5(datas, path_or_obj, name=None, mode='w', key_attrs=None):
    '''
        save datas to HDF5

        Parameters:
            path_or_obj: str, h5py.File or h5py.Group
                hdf5 instance or file name to dump data in

            mode: str 'r+', 'w', 'w-', 'x', or 'a'
                whether create new file or modify exsited
                work only when file name given in `path_or_obj`

                    r+       Read/write, file must exist
                    w        Create file, truncate if exists
                    w- or x  Create file, fail if exists
                    a        Modify if exists, create otherwise

            datas: dict
                data to save to HDF5
                nested structure supported

                key to specify a group or dataset
                val is data is save to dataset or attrs
                    for dataset: 2 types
                        array-like, e.g. np.ndarray: only data
                        tuple `(np.ndarray, dict)` or `(dict,)`
                            dict for attrs set
                    for attrs: dict type

            key_attrs: None, bool, or str
                whether to write attrs in HDF5 object

                if None or bool False, ignore attrs

                if bool True,
                    use default key 'attrs' to fetch value in datas
                        to write in HDF5 as attrs

                if str, that is key in datas
    '''
    assert mode in ['r+', 'w', 'w-', 'x', 'a']

    if isinstance(path_or_obj, (str, bytes)):
        with h5py.File(path_or_obj, mode) as h5f:
            return save_to_hdf5(datas, h5f, name=name, key_attrs=key_attrs)

    grp=path_or_obj
    assert isinstance(grp, (h5py.Group, h5py.File))

    # attrs
    if isinstance(key_attrs, bool):
        key_attrs='attrs' if key_attrs else None

    # datas
    assert isinstance(datas, dict)

    if name is not None:
        datas={name: datas}

    _save_data_to_hdf5_group(grp, datas, key_attrs=key_attrs)

## real work place

### set HDF5 dataset
def _is_append_mode(grp):
    '''
        is append mode for given group
    '''
    return grp.file.mode in ['r+', 'a']

def _save_data_to_hdf5_dataset(grp, name, datas):
    '''
        set data as a Dataset in parent Group

        :param name: str
            name of dataset to save data

        :param datas: array-like, or tuple
            data (and attrs) to set in dataset

            tuple: `(np.ndarray, dict)` or `(dict,)`
    '''
    attrs=None
    if isinstance(datas, tuple) and isinstance(datas[-1], dict):
        if len(datas)==1:
            attrs=datas[0]
            datas=None  # only attrs, no data
        elif len(datas)!=2:
            s=f'only allow 1-/2-tuple for `datas`, but got {len(datas)}-tuple'
            raise ValueError(s)
        else:
            datas, attrs=datas

    if name not in grp:
        if datas is None:
            raise ValueError('no data given to create new Dataset')

        grp.create_dataset(name, data=datas)
    elif datas is not None:
        if name in grp and _is_append_mode(grp):
            del grp[name]
        grp[name]=datas

    dset=grp[name]
    if not isinstance(dset, h5py.Dataset):
        raise ValueError( 'only support to save data to `Dataset,` '
                         f'but got `{type(dset).__name__}`')
    
    # attrs
    if attrs is not None:
        dset.attrs.update(attrs)

### set HDF5 group
def _save_data_to_hdf5_group(grp, datas, key_attrs=None):
    '''
        save data to HDF5 group recursively

        :param key_attrs: None or str
            key in output for attrs from HDF5

            ignore it if attrs is empty or `key_attrs` is None
    '''
    for k, d in datas.items():
        # save data to dataset
        if not isinstance(d, dict):
            _save_data_to_hdf5_dataset(grp, k, d)
            continue

        # attribute
        if k not in grp:
            if key_attrs is not None and k==key_attrs:
                grp.attrs.update(d)
                continue

            grp1=grp.create_group(k)  # create group
        else:
            grp1=grp[k]
            assert isinstance(grp1, h5py.Group)

        # set group recursively
        _save_data_to_hdf5_group(grp1, d, key_attrs=key_attrs)

# cooperate pandas
def save_to_hdf5_pds(pdobjs, path_or_obj, mode='w'):
    '''
        dump pd instances (pd.DataFrame or pd.Series) to HDF5 file

        Parameters:
            filename: str
                file name of HDF5 to dump in

            pdobjs: dict of pd instances
                instances to store
    '''
    assert mode in ['w', 'a']
    if isinstance(path_or_obj, str):   # filename
        with pd.HDFStore(path_or_obj, mode) as store:
            return save_to_hdf5_pds(pdobjs, store)

    store=path_or_obj

    # dump via pd.HDFStore
    for k, p in _squeeze_dict(pdobjs).items():
        assert isinstance(p, (pd.DataFrame, pd.Series))
        store[k]=p

def load_hdf5_pds(path_or_obj, names=None, squeeze=False):
    '''
        load HDF5 to dict of pd instances

        :param names: None or list
            names of pds to load

        :param squeeze: bool, default False
            if True, store all pds in one dict with path in HDF5
            otherwise, use nested dict
    '''
    if isinstance(path_or_obj, str):   # filename
        with pd.HDFStore(path_or_obj, 'r') as store:
            kwargs=dict(squeeze=squeeze, names=names)
            return load_hdf5_pds(store, **kwargs)

    hdfstore=path_or_obj

    if names is None:
        names=list(hdfstore.keys())
    else:
        names=['/'+k if not k.startswith('/') else k for k in names]

    # squeeze
    if squeeze:
        return {k.lstrip('/'): hdfstore[k] for k in names}

    # nested
    res={}
    stores_grp={'': res}  # stores for different groups

    for path, grps, leafs in hdfstore.walk():
        gstore=stores_grp[path]

        # leaves
        for l in leafs:
            k=f'{path}/{l}'
            if k not in names:
                continue

            gstore[l]=hdfstore[k]

        # create store for groups
        for g in grps:
            gstore[g]={}
            stores_grp[f'{path}/{g}']=gstore[g]

    return _del_empty_dict(res)

## auxiliary functions
def _squeeze_dict(data):
    '''
        squeeze dict
    '''
    res={}
    for k, d in data.items():
        k=k.lstrip('/')
        assert k and not k.endswith('/')  # not empty and not ends with '/'

        if isinstance(d, dict):
            for k1, d1 in _squeeze_dict(d).items():
                k2=f'{k}/{k1}'
                assert k2 not in res
                res[k2]=d1

            continue

        assert k not in res
        res[k]=d

    return res

def _del_empty_dict(data):
    '''
        delete empy dict
    '''
    res={}
    for k, d in data.items():
        if isinstance(d, dict):
            if d:
                d=_del_empty_dict(d)

            if not d:  # empty dict
                continue

        res[k]=d

    return res

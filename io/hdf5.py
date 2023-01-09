#!/usr/bin/env python3

'''
    io of HDF5 file
'''

import h5py

import pandas as pd

__all__=['save_to_hdf5', 'load_hdf5',
         'save_pds_to_hdf5', 'load_hdf5_pds']

# dump to hdf5
def save_to_hdf5(datas, path_or_obj, name=None, mode='w', key_attrs=None):
    '''
        save datas to HDF5

        Parameters:
            path_or_obj: str, h5py.File or h5py.Group
                hdf5 instance or file name to dump data in

            mode: str 'w', 'w-', 'x', or 'a'
                whether create new file or modify exsited
                work only when file name given in `path_or_obj`

                    w        Create file, truncate if exists
                    w- or x  Create file, fail if exists
                    a        Modify if exists, create otherwise

            key_attrs: None, bool, or str
                whether to write attrs in HDF5 object

                if None or bool False, ignore attrs

                if bool True,
                    use default key 'attrs' to fetch value in datas
                        to write in HDF5 as attrs

                if str, that is key in datas
    '''
    assert mode in ['w', 'w-', 'x', 'a']

    if isinstance(path_or_obj, (str, bytes)):
        with h5py.File(path_or_obj, mode) as h5f:
            return save_to_hdf5(datas, h5f, name=name, key_attrs=key_attrs)

    grp=path_or_obj

    # attrs
    if isinstance(key_attrs, bool):
        key_attrs='attrs' if key_attrs else None

    # datas
    if name is not None:
        datas={name: datas}
    assert isinstance(datas, dict)

    _dump_dict_hdf5_group_recur(grp, datas, key_attrs=key_attrs)

def load_hdf5(path_or_obj, key_attrs=None):
    '''
        load HDF5 to dict

        Parameters:
            key_attrs: None, bool, or str
                whether to fetch attrs from HDF5 object

                if None or bool False, ignore attrs

                if bool True,
                    use default key 'attrs' for attrs from HDF5
                        in output dict

                if str, that is key in output dict for attrs
    '''
    if isinstance(path_or_obj, (str, bytes)):
        with h5py.File(path_or_obj, 'r') as h5f:
            return load_hdf5(h5f, key_attrs=key_attrs)

    grp=path_or_obj

    # key attrs
    if isinstance(key_attrs, bool):
        key_attrs='attrs' if key_attrs else None

    # fetch from HDF5 recursively
    return _load_dict_hdf5_group_recur(grp, key_attrs=key_attrs)

## real recursive work
def _dump_dict_hdf5_group_recur(grp, datas, key_attrs=None):
    '''
        dump dict to HDF5 group recursively

        :param key_attrs: None or str
            key in output for attrs from HDF5

            ignore it if attrs is empty or `key_attrs` is None
    '''
    for k, d in datas.items():
        # attribute
        if key_attrs is not None and k==key_attrs:
            grp.attrs.update(d)
            continue

        # create group
        if isinstance(d, dict):
            grp1=grp[k] if k in grp else grp.create_group(k)
            _dump_dict_hdf5_group_recur(grp1, d, key_attrs=key_attrs)
            continue

        # dump data
        if k in grp:
            grp[k]=d
        else:
            grp.create_dataset(k, data=d)

def _load_dict_hdf5_group_recur(grp, key_attrs=None):
    '''
        load HDF5 group to dict recursively
    '''
    buffer={}

    for k in grp.keys():
        d=grp[k]

        if isinstance(d, h5py.Group):
            buffer[k]=_load_dict_hdf5_group_recur(d, key_attrs=key_attrs)
            continue

        # buffer[k]=d[:] if d.shape else d[()]
        buffer[k]=d[()]

    # attrs
    if key_attrs is not None and grp.attrs:
        assert key_attrs not in buffer
        buffer[key_attrs]=dict(grp.attrs)

    return buffer

# cooperate pandas
def save_pds_to_hdf5(pdobjs, path_or_obj, mode='w'):
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
            return save_pds_to_hdf5(pdobjs, store)

    store=path_or_obj

    # dump via pd.HDFStore
    for k, p in _squeeze_dict(pdobjs).items():
        assert isinstance(p, (pd.DataFrame, pd.Series))
        store[k]=p

def load_hdf5_pds(path_or_obj, squeeze=False):
    '''
        load HDF5 to dict of pd instances

        :param squeeze: bool, default False
            if True, store all pds in one dict with path in HDF5
            otherwise, use nested dict
    '''
    if isinstance(path_or_obj, str):   # filename
        with pd.HDFStore(path_or_obj, 'r') as store:
            kwargs=dict(squeeze=squeeze)
            return load_hdf5_pds(store, **kwargs)

    hdfstore=path_or_obj

    # squeeze
    if squeeze:
        return {k.lstrip('/'): hdfstore[k] for k in hdfstore.keys()}

    # nested
    res={}
    stores_grp={'': res}  # stores for different groups

    for path, grps, leafs in hdfstore.walk():
        gstore=stores_grp[path]

        # leaves
        for l in leafs:
            gstore[l]=hdfstore[f'{path}/{l}']

        # create store for groups
        for g in grps:
            gstore[g]={}
            stores_grp[f'{path}/{g}']=gstore[g]

    return res

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

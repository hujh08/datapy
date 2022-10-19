#!/usr/bin/env python3

'''
    io of YAML configure file 
'''

import os, shutil

import yaml
import numbers

__all__=['load_yaml', 'save_to_yaml']

# load yaml
def load_yaml(fileobj):
    '''
        load YAML file

        Parameters
            fileobj: str, path object or file-like object
                specify input file

                str or path object: refer to location of the file

                file-like object: like file handle or `StringIO`
    '''

    if not hasattr(fileobj, 'read'):
        with open(fileobj) as f:
            return load_yaml(f)

    return yaml.safe_load(fileobj)

# save dict
def save_to_yaml(data, stream=None, Dumper=None, sort_keys=False,
                       strict_dtype=True, flow_style_nn=True,
                       safe_backup=False,
                       **kwargs):
    '''
        save data to YMAL file

        Parameters:
            stream: str or file-like object with `write` method
                if None, return dumped string

            Dumper: yaml Dumper object
                if None, use `yaml.Dumper`

            sort_keys: bool, default False
                whether to sort keys in dict

            strict_dtype: bool, default True
                if True, use strict data type
                    data would be normalized to following type:
                        mapping data: dict
                        sequece data: list
                        scalar data: str, int, float, complex

            flow_style_nn: bool, default True
                if True, use flow style for non-nested list

                there are 2 styles for collections (seq or mapping)
                    flow or block

            safe_backup: bool, default False
                whether to backup file for safe

                it works only when filename is given as `stream`
    '''
    if Dumper is None:
        Dumper=yaml.Dumper
    kwargs.update(Dumper=Dumper, sort_keys=sort_keys)

    # filename as `stream`
    if stream is not None and not hasattr(stream, 'write'):
        fname=stream
        kwargs.update(strict_dtype=strict_dtype, flow_style_nn=flow_style_nn)

        if not safe_backup:
            with open(fname, 'w') as f:
                return save_to_yaml(data, f, **kwargs)

        # backup for safe
        fname_bkp=fname+'~'
        bkped=False
        succeed=False
        try:
            # backup
            if os.path.isfile(fname):
                shutil.copyfile(fname, fname_bkp)
                bkped=True

            res=save_to_yaml(data, fname, **kwargs)
            succeed=True

            # remove backup if succeed
            if bkped and os.path.isfile(fname):
                os.remove(fname_bkp)

            return res

        except Exception as e:
            if bkped and not succeed:
                os.rename(fname_bkp, fname)

            # re-raise Error
            raise e

    # strict dtype
    if strict_dtype:
        data=_normalize_dtype(data, sort_keys=sort_keys)

    # flow style for non-nested list
    if flow_style_nn:
        Dumper.add_representer(list, _list_representer)

    return yaml.dump(data, stream, **kwargs)

def _list_representer(dumper, data):
    '''
        new representer for List

        use flow style for non-nested list
    '''
    node=dumper.represent_list(data)

    all_scalar=all([isinstance(n, yaml.ScalarNode) for n in node.value])
    if all_scalar:
        node.flow_style=True

    return node

def _normalize_dtype(data, sort_keys=False):
    '''
        normalize data type

        3 classes of data type are distinguished:
            mapping, sequence and scalar

        data type is normalized as following:
            dict-like mapping data: dict
            serialized sequence data: list
            scalar data: str, int, float, complex
    '''

    # scalar
    if isinstance(data, str):
        return str(data)

    if isinstance(data, numbers.Integral):
        return int(data)

    if isinstance(data, numbers.Real):
        return float(data)

    if isinstance(data, numbers.Complex):
        return complex(data)

    # dict
    if isinstance(data, dict):
        keys=data.keys()
        
        if sort_keys:
            keys=sorted(keys)

        result={}
        for k in keys:
            d=_normalize_dtype(data[k], sort_keys=sort_keys)
            k=_normalize_dtype(k)
            result[k]=d

        return result

    # serialized
    return [_normalize_dtype(t, sort_keys=sort_keys) for t in data]

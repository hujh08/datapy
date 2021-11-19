#!/usr/bin/env python3

'''
    io of YAML configure file 
'''

import yaml

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

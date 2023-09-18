#!/usr/bin/env python3

'''
    functions for read/write configure file
'''

import os

from .. import io as dio

class FileConfig:
    '''
        base class for io of configure file

        by default:
            use `dict` as config object
        to support other types of config object,
        overwrite following (static) base methods:
            empty_config() ==> config
            base_get_config(config, key) ==> val
            base_set_config(config, key, val) # inplace set
            base_contains_key(config, key) ==> bool

        to support different file type,
        overwrite following (static) methods:
            base_load_config(fname) ==> config
            base_save_config(config, fname, **kws)
    '''
    def __init__(self, fname, force=False, abspath=False):
        '''
            initial of file config by a filename

            if file not exists or `force` new 
                init with empty config

            Paramters:
                force: bool, default False
                    whether to force using new configure

                abspath: bool, default False
                    whether to use abs path
        '''
        if abspath:
            fname=os.path.abspath(fname)

        self._src=fname
        self._dir_src=os.path.dirname(self._src)

        # config object
        if os.path.exists(self._src) and not force:
            self._config=self.load_config()
        else:
            self._config=self.empty_config()

    # configure object
    @staticmethod
    def empty_config():
        return {}

    @property
    def config(self):
        return self._config

    # load/save configure file

    ## base methods
    @staticmethod
    def base_load_config(fname):
        raise NotImplemented

    @staticmethod
    def base_save_config(config, fname, **kws):
        raise NotImplemented

    ## frequently used
    def load_config(self, fname=None):
        '''
            load config file

            if None given,
                use abspath of source file
                    given when initiating
        '''
        if fname is None:
            fname=self._src

        return self.base_load_config(fname)

    def save_config(self, fname=None, **kws):
        '''
            save config object into a file

            if None given,
                save to source file
                    given when initiating
        '''
        if fname is None:
            fname=self._src

        self.base_save_config(self._config, fname, **kws)

    # get/set config

    ## base methods
    @staticmethod
    def base_get_config(config, key):
        return config[key]

    @staticmethod
    def base_set_config(config, key, val):
        config[key]=val

    def _get_config(self, key):
        return self.base_get_config(self._config, key)

    def _set_config(self, key, val):
        self.base_set_config(self._config, key, val)

    ## config related to file system
    def join_path_to_srcdir(self, path):
        return os.path.join(self._dir_src, path)

    def relpath_to_srcdir(self, path):
        return os.path.relpath(path, self._dir_src)

    def get_path_config(self, key):
        '''
            get configure which is path in file system

            value in configure file
                is related to dir of configure file
        '''
        path=self._get_config(key)
        return self.join_path_to_srcdir(path)

    def set_path_config(self, key, path):
        '''
            set configure which is path in file system

            `path` given is path in file system

            value to set into configure file
                is related to dir of configure file
        '''
        val=self.relpath_to_srcdir(path)
        self._set_config(key, val)

    ## frequently used
    def get_config(self, key, path=False):
        '''
            get configure for given key

            Parameters:
                path: bool, default False
                    whether the configure related to a path
        '''
        if not path:
            return self._get_config(key)
        return self.get_path_config(key)

    def set_config(self, key, val, path=False):
        '''
            set configure for given key

            Parameters:
                path: bool, default False
                    whether the configure related to a path
        '''
        if not path:
            return self._set_config(key, val)
        self.set_path_config(key, val)

    ## magic methods
    def __getitem__(self, key):
        return self.get_config(key)

    def __setitem__(self, key, val):
        self.set_config(key, val)

    # contains key

    ## base methods
    @staticmethod
    def base_contains_key(config, key):
        return key in config

    ## frequently used
    def contains_key(self, key):
        return self.base_contains_key(self._config, key)

    ## magic method
    def __contains__(self, key):
        return self.contains_key(key)

# YAML configure
class YAMLConfig(FileConfig):
    '''
        configure stored in YAML file
    '''

    # load/save configure file
    ## base methods
    @staticmethod
    def base_load_config(fname):
        return dio.load_yaml(fname)

    @staticmethod
    def base_save_config(config, fname, safe_backup=True, **kws):
        kws['safe_backup']=safe_backup
        dio.save_to_yaml(config, fname, **kws)

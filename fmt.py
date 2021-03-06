#!/usr/bin/env python3

'''
class to represent format in a row
'''

import re
import numpy as np

class FmtType:
    '''
    class for single field's type
    '''
    # code for format type
    STR=0
    FLT=1
    INT=2

    CODE={STR: 's', FLT: 'f', INT: 'i'}
    FUNC={STR: str, FLT: float, INT: int}
    # dtype code in numpy
    DTYPE={STR: 'O', FLT: 'f8', INT: 'i8'}

    # reg exp
    INTRE=re.compile(r'[-+]?\d+$')
    FLTRE=re.compile(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$')

    def __init__(self, s):
        # s is in type of string
        if type(s)!=str:
            raise Exception('expect str type, but got %s' % type(s))

        if FmtType.INTRE.match(s):
            self.ftype=FmtType.INT
        elif FmtType.FLTRE.match(s):
            self.ftype=FmtType.FLT
        else:
            self.ftype=FmtType.STR

        self.func=FmtType.FUNC[self.ftype]

    # about numpy
    def dtype(self):
        return FmtType.DTYPE[self.ftype]

    # determine whether this type is str
    def isstr(self):
        return self.ftype==FmtType.STR

    def __call__(self, s):
        return self.func(s)

    # comparison operators
    def __lt__(self, other):
        return self.ftype<other.ftype

    def __eq__(self, other):
        return self.ftype==other.ftype

    # def __gt__(self, other):
    #     return other<self

    def __le__(self, other):
        return self<other or self==other

    # visulization
    def __str__(self):
        return FmtType.CODE[self.ftype]

class Format:
    def __init__(self, fields=None):
        if fields!=None:
            self.fmt=[FmtType(d) for d in fields]

    def update(self, new):
        '''
        according to present Format and format of `fields`,
            update to lower level type

        about level of type of format, see class FmtType
        '''
        if 'fmt' in self.__dict__ and len(new)!=len(self):
            raise Exception('mismatch number of fields')

        if isinstance(new, Format):
            fmt=new.fmt.copy()
        else:
            fmt=[FmtType(d) for d in new]
        if 'fmt' not in self.__dict__:
            self.fmt=fmt
            return

        for i, (new, old) in enumerate(zip(fmt, self.fmt)):
            if old>new:
                self.fmt[i]=new

    # all types is str
    def all_isstr(self):
        for t in self.fmt:
            if not t.isstr():
                return False
        return True

    # type force
    def __call__(self, data):
        '''
        change in-place
        '''
        if len(data)!=len(self):
            raise Exception('mismatch number of fields')

        for i, (d, f) in enumerate(zip(data, self.fmt)):
            data[i]=f(d)

    # visulization
    def __str__(self):
        return ''.join(map(str, self.fmt))

    def __len__(self):
        return len(self.fmt)

    # about numpy
    def formats(self):
        return [t.dtype() for t in self.fmt]

    def dtype(self, names):
        '''
        fmt: format for each column
            works only when names is None
        '''
        return np.dtype({'names': names, 'formats': self.formats()})

    def array(self, data, names):
        '''
        convert 2-d list to numpy array
        '''
        dtype=self.dtype(names)

        data=list(map(tuple, data))
        return np.array(data, dtype=dtype)

    def farray(self, names):
        '''
        return function to convert list
        '''
        return lambda data: self.array(data, names)

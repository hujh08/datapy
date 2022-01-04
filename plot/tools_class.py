#!/usr/bin/env python3

'''
    tools to build class
'''

# proxy methods
def add_proxy_method(dicts, name, target, method_type=None, doc=None):
    '''
        add proxy method with name, `name`
            which is actually passed target object, `target`

        Parameters:
            dicts: dict
                dictionary which the method to add to
                always locals() from calling place

            name: str
                method name

            target: str
                attribute name of object
                    the method is finally passed to

            doc: str, or None
                doc of the method

                if None, use a default one

            method_type: None, 'getter', or 'setter'
                type of the method
    '''
    # construct of method
    # fmt_func="lambda self, {args}: " \
    #          "getattr(getattr(self, '{target}'), '{name}')" \
    #          "({args})"
    fmt_func='lambda self, {args}:  self.{target}.{name}({args})'
    if method_type=='getter':
        args=''
    elif method_type=='setter':
        args='val'
    else:
        assert func is None, \
                'unexpected `method_type`: '+str(method_type)
        args='*args, **kwargs'

    func=eval(fmt_func.format(args=args,
                              name=name, target=target))

    if doc is None:
        doc='proxy of `self.%s.%s`' % (target, name)
    func.__doc__=doc

    # add to dict
    dicts[name]=func

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

# bind new function to existed method
def bind_new_func_to_instance(obj, attr, func, *args, **kwargs):
    '''
        bind new function to an existed method of instance

        Parameters:
            func: callable
                injectable function
                    defined as form
                        def func(obj, oldfunc, oldargs, oldkws, *args, **kwargs)
                    where
                        obj: to passed by instance `obj`
                        oldfunc: passed by backup of original `obj` attr
                        oldargs, oldkws: passed by accepted args inside instance
    '''
    oldfunc=getattr(obj, attr)

    def newfunc(*oldargs, **oldkws):
        return func(obj, oldfunc, oldargs, oldkws, *args, **kwargs)
    newfunc.__doc__=oldfunc.__doc__

    setattr(obj, attr, newfunc)

def bind_new_func_to_instance_by_trans(obj, attr, func):
    '''
        bind to instance method with simple function
            which only transform result from old func (bound method of `obj`)

        Parameters:
            func: callable
                function to handle result from old func

                call by `func(res)`
                    where res from old func
    '''
    def bind_func(obj, oldfunc, oldargs, oldkws):
        res=oldfunc(*oldargs, **oldkws)
        return func(res)

    bind_new_func_to_instance(obj, attr, bind_func)


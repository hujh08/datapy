#!/usr/bin/env python3

'''
    Develop some classes used to parse HTML to extract valueful object

    Parsers are based on BeautifulSoup

    Current tags supported:
        table -- tableParser with result in self.list_table;
        href  -- hrefParser with result in self.dict_href.
'''

import numbers

import pandas as pd
from pandas.io.parsers import TextParser

import bs4
from bs4 import BeautifulSoup

class tableParser:
    '''
        extract <table> tags in html
    '''
    def __init__(self, s=None, **kwargs):
        '''
            init parser with optional string with HTML format
        '''
        self._tables_tags=[]  # tags in tables, multi-level nested list

        if s is not None:
            self.feed(s, **kwargs)

    def feed(self, content, features='lxml', **kwargs):
        '''
            feed in HTML content

            extract tables' tags
        '''
        bs=BeautifulSoup(content, features=features)
        self.feed_bs(bs, **kwargs)

    def feed_bs(self, bs, **kwargs):
        '''
            feed in a BS object with `find_all` method

            optional kwargs is for `BS.find_all`
        '''
        for tab in bs.find_all('table', **kwargs):
            self.feed_tab_tag(tab)

    def feed_tab_tag(self, tag):
        '''
            feed in a BS table tag
        '''
        assert tag.name=='table'

        ctags=[row.find_all('td') for row in tag.find_all('tr')]
        self._tables_tags.append(ctags)

    # getter
    def get_td_tag(self, t, r, c):
        '''
            get a td tag (cell in table)
                in `t`th table, `r`th row and `c`th column
        '''
        return self._tables_tags[t][r][c]

    # to specified data type
    def to_dicts(self, func=None, kwargs_func={}):
        '''
            to nested list of dict

            for each tag of td tag, a dict is obtained

            Parameters:
                func: None or callable
                    if None, use default function
                        `_default_dict_of_td`

                    if callable, would be called by
                        f(tag, trc, **kwargs)
                            where tag is the td tag
                                  trc is index tuple (t, r, c),
                                      table, row and column respectively

                kwargs_func: dict
                    kwargs to `func`
        '''
        if func is None:
            func=self._default_dict_of_td

        res=[]
        for t, tab in enumerate(self._tables_tags):
            res.append([])
            for r, row in enumerate(tab):
                res[-1].append([])
                for c, tag in enumerate(row):
                    res[-1][-1].append(
                        func(tag, (t, r, c), **kwargs_func))
        return res

    def to_dfs(self, head=0, func_td=None, kwargs_td={}, **kwargs):
        '''
            to list of dataframe

            Parameters:
                head: int or list of int
                    index of rows as dataframe head

                func_td, kwargs_td: callable and kwargs
                    used to extract dict from td tag

                    see `.to_dicts` for detail

                kwargs: arguments for `.dicttab_to_df`
                    used to fetch value from obtained dict
                        for head and body respectively

                    see `.dicttab_to_df` for detail
        '''
        tables=self.to_dicts(func=func_td, kwargs_func=kwargs_td)

        kwargs=dict(head=head, **kwargs)
        return [self.dicttab_to_df(t, **kwargs) for t in tables]

    def dicttab_to_df(self, table, head=0, fields=None, naval=None,
                                   keys_head=None, keys_body=None,
                                   **kwargs):
        '''
            convert table with Dict as cell
                to dataframe

            Parameters:
                head: int or list of int
                    index of rows as dataframe head

                fields: None or list
                    columns to return

                    NOTE: it may be different with columns of final dataframe
                        since multi-keys may exist

                    NOTE: it works after `keys_body` normalized

                naval: str, or None
                    value for none-exist key

                    if None, raise Exception
                    if str, return it by default

                keys_head: None, str, list, or dict
                    keys to fetch value as table head
                        given by column-wise

                    if None, use `text` by default
                        which is default key for value, from 'tag.text'

                    if str, key of value to fetch for all columns

                    if list, it is list of str or None
                        if None element, use default key

                        also allow len of list =/= number of columns
                            drop from tail if excessed
                            fill by default if less

                    if dict, key is column index
                        for column not specified, use the default `text`

                        NOTE: column index is initial one

                keys_body: None, str, str tuple, or list of str tuple, dict
                    keys to fetch value as table body
                        given by column-wise

                    if None, use all value in cell Dict

                    if str or str tuple, key(s) of value to fetch for all columns

                    if list, each element for a column
                        if None element, use default key

                        also allow len of list =/= number of columns
                            drop from tail if excessed
                            fill by default if less

                    if dict, key is column index, or head name,
                            the latter is specified by `head` and `keys_head`
                        used in similar way as `keys_head`

                        NOTE: column index is old one

                optional `kwargs`:
                    used in `pandas.io.parsers.readers.TextParser`
        '''
        # input arguments
        ## check dtype
        if len(table)==0:  # empty data
            return pd.DataFrame([])

        ## len of rows
        ncol=len(table[0])
        assert all([len(r)==ncol for r in table[1:]])

        ## head
        if isinstance(head, numbers.Integral):
            head=[head]

        # split head and body
        drows_head=[]  # dicts in rows
        drows_body=[]

        ishead=[False]*len(table)
        for i in head:
            drows_head.append(table[i])
            ishead[i]=True

        for row, h in zip(table, ishead):
            if h:
                continue
            drows_body.append(row)

        # handle table head
        ## normalize keys_head: to list of keys
        keys_head=self._normal_keys_head(keys_head, ncol)

        ## extract value for head
        rhead=[]   # rows of head
        for drow in drows_head:
            rhead.append([])
            for k, d in zip(keys_head, drow):
                rhead[-1].append(self._get_from_dict(d, k, naval=naval))

        chead=list(zip(*rhead))  # list of tuple

        # handle table body
        ## normalize keys_body: to list of tuple of str, or None
        keys_body=self._normal_keys_body(keys_body, ncol, chead)

        ## adjust columns for `fields`
        if fields is not None:
            map_indcol={t: i for i, t in enumerate(chead)}

            tinds=[]
            for t in fields:
                if isinstance(t, str):
                    t=(t,)
                else:
                    assert isinstance(t, tuple), \
                           'only allow tuple and str for `field`'

                tinds.append(map_indcol[t])

            chead=[chead[i] for i in tinds]
            ncol=len(chead)

            keys_body=[keys_body[i] for i in tinds]
            drows_body=[[row[i] for i in tinds] for row in drows_body]

        ## empty body
        if not drows_body:
            if all([len(h)==1 for h in chead]):
                chead=[h[0] for h in chead]
            return pd.DataFrame([], columns=chead)
        nrow_bd=len(drows_body)

        ## extract value for body
        dcols_body=[[] for _ in range(ncol)]  # dicts in columns
        for drow in drows_body:
            for i, d in enumerate(drow):
                dcols_body[i].append(d)

        ckeys=[]  # real keys for columns
        cbody=[]
        for dcol, keys in zip(dcols_body, keys_body):
            keys=self._get_real_body_keys_of_col(keys, dcol)

            col=[]
            for d in dcol:
                col.append(tuple(self._get_from_dict(d, k, naval=naval)
                                    for k in keys))

            ckeys.append(keys)
            cbody.append(col)

        # extend multi-keys
        ismultik=any([len(ks)>1 for ks in ckeys])
        ismixk=False   # mixed keys, both muti and single

        thead_cols=[]   # final table head, in columns
        tbody_cols=[]
        tdfhd_cols=[]   # header of final dataframe
        for chs, cks, cbs in zip(chead, ckeys, cbody):
            if len(cks)>1:
                thead_cols.extend([[*chs, k] for k in cks])
                tdfhd_cols.extend([[*chs, k] for k in cks])

                bcols=[[] for _ in cks]
                for ds in cbs:   # each cell in `cbs` is tuple
                    for i, d in enumerate(ds):
                        bcols[i].append(d)
                tbody_cols.extend(bcols)

            else:
                if len(cks)==1:
                    k=cks[0]
                    bcol=[ds[0] for ds in cbs]
                else: # empty keys, e.g. empty dict in a column and `keys_body=None`
                    k='text'  # use default key
                    bcol=[naval for _ in range(nrow_bd)]

                # head
                if ismultik:
                    thead_cols.append([*chs, k])
                    tdfhd_cols.append([*chs, ''])  # not allow '' in TextParser
                    ismixk=True
                else:
                    thead_cols.append(list(chs))
                    tdfhd_cols.append(list(chs))

                # body
                tbody_cols.append(bcol)

        # to dataframe
        nrow_hd=len(thead_cols[0])  # num of rows for table head

        nrow=nrow_hd+nrow_bd
        trows=[[] for _ in range(nrow)]  # table rows
        for i, (hd, bd) in enumerate(zip(thead_cols, tbody_cols)):
            for i, t in enumerate(hd+bd):
                trows[i].append(t)

        if naval is not None:
            assert 'na_values' not in kwargs
            kwargs['na_values']=naval

        with TextParser(trows, header=list(range(nrow_hd)), **kwargs) as p:
            # also infer data type automatically in `TextPaser`
            df=p.read()

        ## reset df columns
        if nrow_hd==1:
            df.columns=pd.Index([t[0] for t in tdfhd_cols])
        elif ismixk:  # reset df columns if keys is of mixed len
            df.columns=pd.MultiIndex.from_tuples(tdfhd_cols)

        return df

    ## functions to handle td tag
    def _default_dict_of_td(self, tag, trc, **kwargs):
        '''
            default extractor of td

            extract `text` and `hrefs` by default

            dict is returned
        '''
        data={'text': tag.text}

        hrefs=self._ext_hrefs_from_tag(tag, **kwargs)
        if len(hrefs)==1:
            k=list(hrefs.keys())[0]
            data['href']=hrefs[k]
        else:
            for k in hrefs:
                s='_'.join(k.split())
                data['href_'+s]=hrefs[k]

        return data

    ## auxiliary functions
    @staticmethod
    def _ext_hrefs_from_tag(tag, key=None, **kwargs):
        '''
            extract hrefs from tag
        '''
        p=hrefParser()
        p.feed_bs(tag, **kwargs)

        return p.to_dict(key=None)

    @staticmethod
    def _get_from_dict(d, key, naval=None):
        '''
            fetch value from dict
                for given `key`

            :param `naval`: None or str
                work for non-exists key

                if None, raise exception
                otherwise return `naval` by default
        '''
        if key in d:
            return d[key]

        if naval is None:
            raise KeyError('key not exists: %s' % key)

        return naval

    @staticmethod
    def _normal_keys_head(keys_head, ncol):
        '''
            normalize `keys_head`
                keys to fetch value as table head

            return list of str
                each as key for a column

            SUPPORT: None, str, list, or dict
                if None, use `text` by default
                    which is default key for value, from 'tag.text'

                if str, key of value to fetch for all columns

                if list, it is list of str or None
                    for None, use default key

                    also allow len of list =/= number of columns
                        drop from tail if excessed
                        fill by default if less

                if dict, key is column index
                    for column not specified, use the default `text`
        '''
        key_default='text'
        if keys_head is None:
            res=[key_default]*ncol
        elif isinstance(keys_head, str):
            res=[keys_head]*ncol
        elif isinstance(keys_head, dict):
            res=[keys_head.get(i, key_default) for i in range(ncol)]
        elif isinstance(keys_head, list):
            res=[key_default if t is None else t for t in keys_head[:ncol]]
            res=res+([key_default]*(ncol-len(res)))
        else:
            raise TypeError('unsupported type for `keys_head`: %s'
                                % type(keys_head).__name__)

        return res

    @staticmethod
    def _normal_keys_body(keys_body, ncol, chead):
        '''
            normalize `keys_body`
                keys to fetch value as table boyd

            return list of str tuple or None
                each as keys for a column

            :param `chead`: already obtained table head
                list of tuple, each for a column

            SUPPORT: None, str, str tuple, or list of str tuple, dict
                keys to fetch value as table body
                    given by column-wise

                if None, use all values in cell Dict

                if str or str tuple, key(s) of value to fetch for all columns

                if list, each element for a column
                    if None element, use default key

                    also allow len of list =/= number of columns
                        drop from tail if excessed
                        fill by default if less

                if dict, key is column index, or head name
                    given in `chead`

                    allow striped name if len of name tuple is 1
        '''
        key_default=None   # default key

        if keys_body is None:
            res=[key_default]*ncol
        elif isinstance(keys_body, str):
            res=[(keys_body,)]*ncol
        elif isinstance(keys_body, tuple):
            assert all([isinstance(t, str) for t in keys_body])
            res=[tuple(keys_body)]*ncol
        else:
            if isinstance(keys_body, list):
                keys_body=dict(enumerate(keys_body))

            if not isinstance(keys_body, dict):
                raise TypeError('unsupported type for `keys_body`: %s'
                                % type(keys_body).__name__)

            res=[]
            for i in range(ncol):
                if i in keys_body:
                    t=keys_body[i]
                else:
                    name=chead[i]
                    assert isinstance(name, tuple)

                    if name in keys_body:
                        t=keys_body[name]
                    elif len(name)==1 and name[0] in keys_body:
                        # strip out of tuple
                        t=keys_body[name[0]]
                    else:
                        t=key_default

                # normalize key to name or str tuple
                if isinstance(t, str):
                    t=(t,)
                elif t!=key_default:
                    assert isinstance(t, tuple)

                res.append(t)

            res=res+([key_default]*(ncol-len(res)))

        return res

    @staticmethod
    def _get_real_body_keys_of_col(keys, dcol):
        '''
            get real keys for dict in a column of table body

            mainly handle some special keys, that are
                None: use all keys in data
        '''
        if keys is None:  # use all keys if key is None
            keys=set()
            for d in dcol:
                keys.update(d.keys())
            return tuple(keys)

        return keys
    
class hrefParser:
    '''
        extract <a> tags in html
    '''
    def __init__(self, s=None, **kwargs):
        '''
            init parser with optional string with HTML format
        '''
        self._hrefs_tags=[]  # href tags

        if s is not None:
            self.feed(s, **kwargs)

    def feed(self, content, features='lxml', **kwargs):
        '''
            feed in HTML content

            extract hrefs' tags
        '''
        bs=BeautifulSoup(content, features=features)
        self.feed_bs(bs, **kwargs)

    def feed_bs(self, bs, **kwargs):
        '''
            feed in a BS object with `find_all` method

            optional kwargs is for `BS.find_all`
        '''
        self._hrefs_tags.extend(bs.find_all('a', **kwargs))

    # to specified data type
    def to_dict(self, key=None):
        '''
            to dict of hrefs

            Parameters:
                key: None or callable
                    function to get key of dict
                        called by `f(tag)

                    if None, use default function
                        just return tag.text

                    if callable, would be called by
                        f(tag, i)
                            where tag is the <a> tag
                                  i is index of tag,
                                       also order to get it
        '''
        if key is None:
            key=lambda tag, i: tag.text

        hrefs=[(key(t, i), t.attrs['href'])
                    for i, t in enumerate(self._hrefs_tags)]

        # unique key
        cnt={}
        for k, _ in hrefs:
            if k not in cnt:
                cnt[k]=1
            else:
                cnt[k]+=1

        uniq={}
        for k, n in cnt.items():
            if n==1:
                uniq[k]=[k]
            else:
                i=0
                uniq[k]=[]
                for _ in range(n):
                    for _ in range(len(hrefs)):  # maybe e.g. k0 already exists
                        s='%s%i' % (k, i)
                        i+=1
                        if s not in cnt:
                            break
                    uniq[k].append(s)

        # dict
        res={}
        for k, v in hrefs:
            s=uniq[k].pop(0)
            res[s]=v

        return res


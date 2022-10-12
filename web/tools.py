#!/usr/bin/env python3

'''
    simple tools to handle HTML

    other more complicated parsers are designed in `parser.py`
'''

import os

import pandas as pd

from bs4 import BeautifulSoup

# to dataframes
def ext_dfs(page):
    '''
        extract tables in HTML page as dataframes 
    '''
    return pd.read_html(page)

# extract hrefs
def ext_hrefs(page, return_dict=False, features='lxml', **kwargs):
    '''
        extract hrefs in HTML page

        return list of urls by default

        if `return_dict` is True,
            use tag.text as key

        optional `kwargs` used to locate tag
            passed to `bs.find_all`
    '''
    # feed in filename of HTML file
    if os.path.isfile(page):
        with open(page) as f:
            kwargs.update(return_dict=return_dict, features=features)
            return ext_hrefs(f.read(), **kwargs)

    # feed in content of page
    bs=BeautifulSoup(page, features=features)

    tags=bs.find_all('a', **kwargs)

    if not return_dict:
        return [t.attrs['href'] for t in tags]

    # return dict
    res={}
    uniq_keys={}
    for tag in tags:
        key=tag.text
        href=tag.attrs['href']

        if key in res:
            key0=key

            i=uniq_keys[key0]+1
            key=key0+' (%i)' % i
            while key in res:
                i+=1
                key=key0+' (%i)' % i

            uniq_keys[key0]=i

        else:
            uniq_keys[key]=1

        res[key]=href

    return res

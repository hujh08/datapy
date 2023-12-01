#!/usr/bin/env python3

'''
    utilities to handle text file
'''

import os
import re

# remove comment of line
def line_comment_strip(line, comment='#'):
    '''
        remove comment in a line
    '''
    assert len(comment)==1  # only support single char

    return re.sub(r'[%s].*$' % comment, '', line)

# strip empty lines from head or tail
def strip_empty_lines(doc, join_lines=True):
    '''
        strip head and tail empty lines
    '''
    lines=[]

    skip_empty_head=True
    for line in doc.splitlines():
        if skip_empty_head:
            if not line:
                continue
            skip_empty_head=False

        if not line:
            break

        lines.append(line)

    if not join_lines:
        return lines

    return os.linesep.join(lines)

# extract word in a line
def words_in_line(line):
    '''
        extract words in a line
        return
            (words, spans)
            where
                words: list of word
                spans: list of pan for each word
    '''
    p=re.compile(r'\b\w+\b')
    matches=[(t.span(), t.group()) for t in p.finditer(line)]
    spans, words=zip(*matches)

    return words, spans

#!/usr/bin/env python3

'''
    utilities to handle text file
'''

import os
import re

# read nth line
def read_nth_line(fileobj, n, restore_stream=False):
    '''
        read nth line

        `n` is 0-indexed

        `restore_stream`: bool
            whether to restore stream position

            if False, after return, current position would be (n+1)-th line
    '''
    if isinstance(fileobj, str):
        with open(fileobj) as f:
            return read_nth_line(f, n)

    assert n>=0
    if restore_stream:
        t=fileobj.tell()

    for _ in range(n+1):
        line=fileobj.readline()

    if restore_stream:
        fileobj.seek(t)

    return line

# remove comment of line
def rstrip_line_comment(line, comment='#'):
    '''
        remove comment in a line from end
            right strip
    '''
    return re.sub(r'[%s].*$' % comment, '', line)

def lstrip_line_comment_chars(line, comment='#'):
    '''
        remove comment chars in head of line
    '''
    return re.sub(r'^[%s\s]*' % comment, '', line)

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
def words_in_line(line, only_word_char=False):
    '''
        extract words in a line
        return
            (words, spans)
            where
                words: list of word
                spans: list of pan for each word
                    each span (t0, t1)
                        where line[t0:t1] is the word

        :param only_word_char: bool, default False
            whether only allow results with word chars
                that is alphanumeric chars and underscore (_)
                    '\\w' in `re`

            if False,
                split words by whitespace
    '''
    if only_word_char:
        p=re.compile(r'\b\w+\b')
    else:
        p=re.compile(r'(^|(?<=\s))\S+(?=\s|$)')
    matches=[(t.span(), t.group()) for t in p.finditer(line)]
    spans, words=zip(*matches)

    return words, spans

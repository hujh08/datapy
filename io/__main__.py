#!/usr/bin/env python3

'''
    command line interface to handle data file
'''

from ._main_parser import parser

# run command
args=parser.parse_args()

args.func(args)

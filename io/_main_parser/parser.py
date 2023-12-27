
'''
    main script
'''

from .util import new_parser, import_local_modules

# import subparser constuctors
modules=['parser_yaml']
constructors=import_local_modules(modules)

# argument parser
parser=new_parser()

## subparsers
subparsers=parser.add_subparsers(help='sub-commands')

### constructors of parsers
for p in constructors:
    p.create_parser(parent=subparsers)

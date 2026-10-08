"""Maintain classified planning build artifacts: build, status or clean."""

import argparse
import json

from .maintenance import build_native, clean_native, native_status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build','status','clean'))
    args=parser.parse_args()
    if args.command == 'build':
        result=build_native()
    elif args.command == 'clean':
        result={'removed_native_cache':clean_native()}
    else:
        result=native_status()
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

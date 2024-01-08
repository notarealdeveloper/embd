#!/usr/bin/env python3

__all__ = ['main']

import os
import sys
import argparse
import embd


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser('think')
    parser.add_argument('file', nargs='?')
    parser.add_argument('-l', '--lines', action='store_true')
    parser.add_argument('-c', '--cat', action='store_true')
    parser.add_argument('-A', '--after', type=int, default=0)
    parser.add_argument('-B', '--before', type=int, default=0)
    parser.add_argument('-C', '--context', type=int, default=0)
    args = parser.parse_args(argv)

    before = after = 0
    if args.context:
        before = after = args.context
    if args.before:
        before = args.before
    if args.after:
        after = args.after

    if before > 0 or after > 0:
        # context arguments imply windowed embeddings, so turn on -l
        args.lines = True

    if not args.file:
        input = sys.stdin.read()
    else:
        input = open(args.file).read()

    if args.lines:
        texts = input.splitlines()
    else:
        texts = [input]

    if args.cat:
        texts = [open(text, 'r').read() for text in texts]

    space = embd.Space()

    outputs = []
    for n in range(len(texts)):
        start = max(0, n-before)
        end   = n+after+1
        window = texts[start:end]
        if len(window) == 0:
            continue
        text = '\n'.join(window)
        output = space.think(text)
        outputs.append(output)

    import numpy as np
    tensor = np.stack(outputs)

    if os.isatty(sys.stdout.fileno()):
        print(tensor)
    else:
        bytes = embd.tensor_to_bytes(tensor)
        sys.stdout.buffer.write(bytes)

if __name__ == '__main__':
    main(sys.argv[1:])

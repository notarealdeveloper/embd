#!/usr/bin/env python3

__all__ = ['main']

import os
import sys
import argparse
import multiprocessing
import embd

_recursive_space = None


def iter_files(paths):
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs.sort()
                for file in sorted(files):
                    file = os.path.join(root, file)
                    if os.path.isfile(file):
                        yield file
        elif os.path.isfile(path):
            yield path
        else:
            raise FileNotFoundError(path)


def init_recursive_worker():
    global _recursive_space
    _recursive_space = embd.Space()


def embed_file(path):
    _recursive_space.get_from_path(path)
    return path


def embed_files(paths, processes=None):
    files = list(dict.fromkeys(iter_files(paths)))
    if not files:
        return []

    processes = processes or os.cpu_count() or 1
    processes = min(processes, len(files))
    with multiprocessing.Pool(
        processes=processes,
        initializer=init_recursive_worker,
    ) as pool:
        return list(pool.imap_unordered(embed_file, files))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser('think')
    parser.add_argument('file', nargs='*')
    parser.add_argument('-l', '--lines', action='store_true')
    parser.add_argument('-c', '--cat', action='store_true')
    parser.add_argument('-r', '--recursive', action='store_true')
    parser.add_argument('-A', '--after', type=int, default=0)
    parser.add_argument('-B', '--before', type=int, default=0)
    parser.add_argument('-C', '--context', type=int, default=0)
    args = parser.parse_args(argv)

    if args.recursive:
        if args.lines or args.cat or args.before or args.after or args.context:
            parser.error('-r/--recursive cannot be combined with line, cat, or context options')
        if not args.file:
            parser.error('-r/--recursive requires at least one file or directory')
        try:
            embed_files(args.file)
        except OSError as e:
            parser.error(str(e))
        return

    if len(args.file) > 1:
        parser.error('multiple files require -r/--recursive')

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
        input = open(args.file[0]).read()

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

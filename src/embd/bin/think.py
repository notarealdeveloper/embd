#!/usr/bin/env python3

__all__ = ['main']

import os
import sys
import argparse
import multiprocessing
import subprocess
import embd

_recursive_space = None

MAX_DATA_BYTES = 500 * 1024
MAX_WEIRD_BYTES = 64 * 1024
BINARY_MIME_PREFIXES = (
    'audio/',
    'font/',
    'image/',
    'video/',
)
BINARY_MIME_TYPES = {
    'application/gzip',
    'application/java-archive',
    'application/pdf',
    'application/vnd.ms-cab-compressed',
    'application/vnd.rar',
    'application/x-7z-compressed',
    'application/x-archive',
    'application/x-bzip2',
    'application/x-cpio',
    'application/x-dosexec',
    'application/x-elf',
    'application/x-executable',
    'application/x-mach-binary',
    'application/x-msdownload',
    'application/x-object',
    'application/x-pie-executable',
    'application/x-sharedlib',
    'application/x-tar',
    'application/x-xz',
    'application/zip',
}
TEXT_MIME_TYPES = {
    'application/csv',
    'application/ecmascript',
    'application/javascript',
    'application/json',
    'application/ld+json',
    'application/sql',
    'application/toml',
    'application/xml',
    'application/x-awk',
    'application/x-empty',
    'application/x-httpd-php',
    'application/x-perl',
    'application/x-python',
    'application/x-ruby',
    'application/x-shellscript',
    'application/x-yaml',
}
TEXT_DESCRIPTION_WORDS = (
    'ascii text',
    'csv',
    'empty',
    'json',
    'script',
    'text',
    'unicode text',
    'utf-8',
    'xml',
)


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


def weird_file_budget(file_count):
    return min(25, max(3, file_count // 50))


def run_file(path, *args):
    try:
        result = subprocess.run(
            ['file', *args, path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        return ''
    return result.stdout.strip().lower()


def file_type(path):
    return (
        run_file(path, '-b', '--mime-type'),
        run_file(path, '-b'),
    )


def is_text_mime(mime_type):
    return mime_type.startswith('text/') or mime_type in TEXT_MIME_TYPES


def is_binary_mime(mime_type):
    return (
        any(mime_type.startswith(prefix) for prefix in BINARY_MIME_PREFIXES)
        or mime_type in BINARY_MIME_TYPES
    )


def is_text_description(description):
    return any(word in description for word in TEXT_DESCRIPTION_WORDS)


def fallback_text_probe(path):
    try:
        with open(path, 'rb') as file:
            data = file.read(MAX_WEIRD_BYTES + 1)
    except OSError:
        return False

    if b'\0' in data:
        return False
    try:
        data.decode('utf-8')
    except UnicodeDecodeError:
        return False
    return True


def should_embed_file(path, weird_files_allowed=0):
    size = os.path.getsize(path)
    mime_type, description = file_type(path)

    if mime_type and is_text_mime(mime_type):
        return True, False

    if description and is_text_description(description):
        return True, False

    if size > MAX_DATA_BYTES:
        return False, False

    if mime_type and is_binary_mime(mime_type):
        return False, False

    if size <= MAX_WEIRD_BYTES and weird_files_allowed > 0:
        if not mime_type and fallback_text_probe(path):
            return True, True
        if mime_type not in BINARY_MIME_TYPES:
            return True, True

    return False, False


def filter_embeddable_files(files):
    weird_files_allowed = weird_file_budget(len(files))
    embeddable = []
    for file in files:
        should_embed, used_weird_allowance = should_embed_file(file, weird_files_allowed)
        if should_embed:
            embeddable.append(file)
        if used_weird_allowance:
            weird_files_allowed -= 1
    return embeddable


def init_recursive_worker():
    global _recursive_space
    _recursive_space = embd.Space()


def embed_file(path):
    _recursive_space.get_from_path(path)
    return path


def embed_files(paths, processes=None, force=False):
    files = list(dict.fromkeys(iter_files(paths)))
    if not force:
        files = filter_embeddable_files(files)
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
    parser.add_argument('-f', '--force', action='store_true')
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
            embed_files(args.file, force=args.force)
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

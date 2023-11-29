""" space
    =====
    compute embeddings that are
    * automatically cached by content
    * persisted between processes 
    * namespaced by model and args
    * supports model independent name assignment
"""

__all__ = [
    'Space',
    'think',
    'load_name',
    'save_name',
    'load_blob',
    'save_blob',
    'get',
    'get_from_blob',
    'get_from_path',
    'get_from_name_and_path',
    'get_from_name_and_blob',
]

import numpy as np
from functools import lru_cache

from mmry import Cache
from .embed import Embed
from .bytes import bytes_to_tensor, tensor_to_bytes

def think(text):
    return Space.default().think(text)

def load_name(name):
    return Space.default().load_name(name)

def save_name(name, blob):
    return Space.default().save_name(name, blob)

def load_blob(blob):
    return Space.default().load_blob(blob)

def save_blob(blob, embed):
    return Space.default().save_blob(blob, embed)

def get(text=None, *, path=None, name=None):
    return Space.default().get(text=text, path=path, name=name)

def get_from_blob(blob):
    return Space.default().get_from_blob(blob)

def get_from_path(path):
    return Space.default().get_from_path(path)

def get_from_name_and_path(name, path):
    return Space.default().get_from_name_and_path(name, path)

def get_from_name_and_blob(name, blob):
    return Space.default().get_from_name_and_blob(name, blob)

class Space:

    """
        An embedding space
    """

    def __init__(self, embed=None):
        self.embed = embed or Embed()
        self.name = self.embed.namespace()
        self.cache = Cache(self.name)
        assert self.name == self.cache.namespace()
        assert self.name == self.embed.namespace()

    def think(self, arg):

        if isinstance(arg, (str, bytes)):
            return self.get_from_blob(arg)

        if isinstance(arg, (list, tuple, set)):
            return np.stack([self.think(text) for text in arg])

        if isinstance(arg, dict):
            keys = arg.keys()
            vals = arg.values()
            embs = [self.think(val) for val in vals]
            return dict(zip(keys, embs))

        raise TypeError(f"Can't think about {arg.__class__.__name__}: {arg!r}")

    def __call__(self, text):
        return self.think(text)

    def save_blob(self, blob, embed):
        bytes = tensor_to_bytes(embed)
        self.cache.save_blob(blob, bytes)

    @lru_cache(maxsize=None)
    def load_blob(self, blob):
        bytes = self.cache.load_blob(blob)
        return bytes_to_tensor(bytes)

    def save_name(self, name, blob):
        try: return self.cache.save_name(name, blob)
        except FileExistsError: pass

    @lru_cache(maxsize=None)
    def load_name(self, name):
        bytes = self.cache.load_name(name)
        return bytes_to_tensor(bytes)

    def get(self, text=None, *, path=None, name=None):
        if path is None and text is None:
            raise ValueError(f"at least one of path and text must be specified")
        if path is not None and text is not None:
            raise ValueError(f"at most one of path and text may be specified")
        if path is not None and name is None:
            return self.get_from_path(path)
        if path is not None and name is not None:
            return self.get_from_name_and_path(name, path)
        if path is None and name is None:
            return self.get_from_blob(text)
        if path is None and name is not None:
            return self.get_from_name_and_blob(name, text)
        raise RuntimeError(f"Impossible situation reached")

    def get_from_blob(self, blob):
        """ load or create embedding from text or bytes """
        try:
            return self.load_blob(blob)
        except:
            embed = self.embed(blob)
            self.save_blob(blob, embed)
            return embed

    def get_from_path(self, path):
        """ load or create embedding from file """
        blob = open(path).read()
        return self.get_from_blob(blob)

    def get_from_name_and_path(self, name, path):
        """ load or create embedding from name or file """
        try:
            return self.load_name(name)
        except:
            blob = open(path).read()
            embed = self.get_from_blob(blob)
            self.save_name(name, blob)
        return embed

    def get_from_name_and_blob(self, name, blob):
        """ load or create embedding from name or file """
        try:
            return self.load_name(name)
        except:
            embed = self.get_from_blob(blob)
            self.save_name(name, blob)
        return embed

    @classmethod
    def default(cls):
        try:
            return cls._default
        except:
            cls._default = cls()
            return cls._default

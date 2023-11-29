""" space
    =====
    compute embeddings that are
    * automatically cached by content
    * persisted between processes 
    * namespaced by model and args
    * supports model independent name assignment
"""

__all__ = [
    'think',
    'Space',
]

import numpy as np
from functools import lru_cache

from mmry import Cache
from .embed import Embed
from .bytes import bytes_to_tensor, tensor_to_bytes

def think(text, *, name=None):
    space = Space.default()
    return space.think(text, name=name)

class Space:

    """ An embedding space """

    def __init__(self, embed=None):
        self.embed = embed or Embed()
        self.name = self.embed.namespace()
        self.cache = Cache(self.name)
        assert self.name == self.cache.namespace()
        assert self.name == self.embed.namespace()

    def think(self, arg, *, name=None):
        if isinstance(arg, (str, bytes)):
            return self.load_or_create(arg, name=name)

        # no support at present for loading collections by name
        assert name is None

        if isinstance(arg, (list, tuple, set)):
            return np.stack([self.think(text) for text in arg])

        if isinstance(arg, dict):
            keys = arg.keys()
            vals = arg.values()
            embs = [self.think(val) for val in vals]
            return dict(zip(keys, embs))

        raise TypeError(f"Can't think about {arg.__class__.__name__}: {arg!r}")

    def __call__(self, text, *, name=None):
        return self.think(text, name=name)

    def load_or_create(self, blob, *, name=None):
        if name is not None:
            return self.load_or_create_name(blob, name)
        else:
            return self.load_or_create_blob(blob)

    def load_or_create_blob(self, blob):
        try:
            return self.load_blob(blob)
        except:
            embed = self.embed(blob)
            self.save_blob(blob, embed)
            return embed

    def load_or_create_name(self, blob, name):
        try:
            return self.load_name(name)
        except:
            embed = self.load_or_create_blob(blob)
            self.save_name(name, blob)
            return embed

    def save_blob(self, blob, embed):
        bytes = tensor_to_bytes(embed)
        self.cache.save_blob(blob, bytes)

    def load_blob(self, blob):
        bytes = self.cache.load_blob(blob)
        return bytes_to_tensor(bytes)

    def save_name(self, name, blob):
        try: return self.cache.save_name(name, blob)
        except FileExistsError: pass

    def load_name(self, name):
        return self.cache.load_name(name)

    @classmethod
    def default(cls):
        try:
            return cls._default
        except:
            cls._default = cls()
            return cls._default

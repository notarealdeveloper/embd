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

def think(arg):
    space = Space.default()
    return space.think(arg)

class Space:

    """ An embedding space """

    def __init__(self, embed=None):
        self.embed = embed or Embed()
        self.name = self.embed.namespace()
        self.cache = Cache(self.name)
        assert self.name == self.cache.namespace()
        assert self.name == self.embed.namespace()

    def think(self, arg):
        if isinstance(arg, (str, bytes)):
            return self.get(arg)
        if isinstance(arg, (list, tuple, set)):
            return np.stack([self.get(a) for a in arg])
        if isinstance(arg, dict):
            keys = arg.keys()
            vals = arg.values()
            embs = [self.get(val) for val in vals]
            return dict(zip(keys, embs))
        raise TypeError(f"Can't think about {arg.__class__.__name__}: {arg!r}")

    def __call__(self, arg):
        return self.think(arg)

    def gets(self, blobs):
        """ Not currently used, but keep this around in case
            we want to enable it for speeding up the GPU case.
            In all current uses it seemed to slow things down.
        """
        raise NotImplementedError("Don't use this yet")
        embeds = {}
        todos = {}
        for n, blob in enumerate(blobs):
            try:
                embeds[n] = self.load(blob)
            except:
                todos[n] = blob
        ns = list(todos.keys())
        bs = list(todos.values())
        es = self.embed(bs)
        for n, b, e in zip(ns, bs, es):
            embeds[n] = e
            self.save(b, e)
        return np.stack([e for n,e in sorted(embeds.items())])

    def get(self, blob):
        try:
            embed = self.load(blob)
            return embed
        except:
            embed = self.embed(blob)
            self.save(blob, embed)
            return embed

    def save(self, blob, embed):
        bytes = tensor_to_bytes(embed)
        self.cache.save_blob(blob, bytes)

    def load(self, blob):
        bytes = self.cache.load_blob(blob)
        return bytes_to_tensor(bytes)

    @classmethod
    def default(cls):
        try:
            return cls._default
        except:
            cls._default = cls()
            return cls._default


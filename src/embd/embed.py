"""
Turn software 1.0 types (e.g., text)
into software 2.0 types (e.g., vect)

The functions here are implemented as classes,
for the same reasons that neural networks are.
"""

__all__ = [
    'EmbedDefault',
    'EmbedFlag',
]

import os

class EmbedFlag:

    from functools import lru_cache

    SIZES = {'small': 384, 'base': 768, 'large': 1024}

    @classmethod
    @lru_cache(maxsize=1)
    def load_model(cls, name):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name)
        return model

    def __init__(self, size='large', normalized=True):
        if size not in self.SIZES:
            raise ValueError(f"size must be one of: {self.SIZES}")
        self.size = size
        self.normalized = normalized
        self.model_name = f"BAAI/bge-{size}-en-v1.5"

    def shape(self):
        return self.SIZES[self.size]

    def namespace(self):
        normalized = 'normalized' if self.normalized else 'unnormalized'
        return f"embed/flag/{self.size}/{normalized}"

    @property
    def model(self):
        # lazily load model
        try:
            return self._model
        except:
            self._model = self.load_model(self.model_name)
            return self._model

    def __call__(self, text):
        # the large flag model has a crazy non-constant embedding size
        # depending on whether or not we feed it a token length string
        return self.model.encode(text, normalize_embeddings=self.normalized)

EmbedDefault = EmbedFlag

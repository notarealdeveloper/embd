"""
Turn software 1.0 types (e.g., text)
into software 2.0 types (e.g., vect)

The functions here are implemented as classes,
for the same reasons that neural networks are.
"""

__all__ = [
    'Embed',
    'EmbedFlag',
    'EmbedMpnet',
    'EmbedMinilm',
    'EmbedGtrT5',
    'EmbedSentenceT5',
    'EmbedFlagLLMEmbedder',
    'EmbedFlagReranker',
    'list_models',
    'get_default_model',
    'set_default_model',
]

class EmbedBase:

    from functools import lru_cache

    @classmethod
    @lru_cache(maxsize=1)
    def load_model(cls, name):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name)
        return model

    @property
    def model(self):
        # lazily load model
        try:
            return self._model
        except:
            self._model = self.load_model(self.model_name)
            return self._model

class EmbedFlag(EmbedBase):

    SIZES = {'small': 384, 'base': 768, 'large': 1024}

    def __init__(self, size='large', normalized=True):
        if size not in self.SIZES:
            raise ValueError(f"size must be one of: {self.SIZES}")
        self.size = size
        self.normalized = normalized
        self.model_name = f"BAAI/bge-{size}-en-v1.5"

    def namespace(self):
        normalized = 'normalized' if self.normalized else 'unnormalized'
        return f"embed/flag/{self.size}/{normalized}"

    def shape(self):
        return self.SIZES[self.size]

    def __call__(self, text):
        return self.model.encode(text, normalize_embeddings=self.normalized)


class EmbedFlagLLMEmbedder(EmbedBase):

    """ paper: Retrieve Anything To Augment Large Language Models
        url: https://arxiv.org/abs/2310.07554
    """
    def __init__(self, normalized=True):
        self.normalized = normalized
        self.model_name = f"BAAI/llm-embedder"

    def namespace(self):
        normalized = 'normalized' if self.normalized else 'unnormalized'
        return f"embed/flag-llm-embedder/{normalized}"

    def shape(self):
        return 768

    def __call__(self, text):
        return self.model.encode(text)


class EmbedFlagReranker(EmbedBase):

    SIZES = {'base': 768, 'large': 1024}

    def __init__(self, size='large', normalized=True):
        if size not in self.SIZES:
            raise ValueError(f"size must be one of: {self.SIZES}")
        self.size = size
        self.normalized = normalized
        self.model_name = f"BAAI/bge-reranker-{size}"

    def namespace(self):
        normalized = 'normalized' if self.normalized else 'unnormalized'
        return f"embed/flag-reranker/{self.size}/{normalized}"

    def shape(self):
        return 'Unknown'

    def __call__(self, text):
        return self.model.encode(text)



class EmbedMpnet(EmbedBase):

    SIZES = {'base': 768}

    def __init__(self, size='base'):
        if size not in self.SIZES:
            raise ValueError(f"size must be one of: {self.SIZES}")
        self.model_name = f"sentence-transformers/all-mpnet-{size}-v2"
        self.size = size

    def namespace(self):
        return f"embed/mpnet/{self.size}"

    def shape(self):
        return self.SIZES[self.size]

    def __call__(self, text):
        return self.model.encode(text)


class EmbedMinilm(EmbedBase):

    SIZES    = {'base': 384, 'large': 384}
    VERSIONS = {'base': 'L6', 'large': 'L12'}

    def __init__(self, size='base'):
        if size not in self.SIZES:
            raise ValueError(f"size must be one of: {self.SIZES}")
        version = self.VERSIONS[size]
        self.model_name = f"sentence-transformers/all-MiniLM-{version}-v2"
        self.size = size

    def namespace(self):
        return f"embed/minilm/{self.size}"

    def shape(self):
        return self.SIZES[self.size]

    def __call__(self, text):
        return self.model.encode(text)

class EmbedGtrT5(EmbedBase):

    SIZES = {'base': 768, 'large': 768, 'xl': 768, 'xxl': 768}

    def __init__(self, size='xl'):
        if size not in self.SIZES:
            raise ValueError(f"size must be one of: {self.SIZES}")
        self.size = size
        self.model_name = f"sentence-transformers/gtr-t5-{size}"

    def namespace(self):
        return f"embed/gtr-t5/{self.size}"

    def shape(self):
        return self.SIZES[self.size]

    def __call__(self, text):
        return self.model.encode(text)

class EmbedSentenceT5(EmbedBase):

    SIZES = {'base': 768, 'large': 768, 'xl': 768, 'xxl': 768}

    def __init__(self, size='xl'):
        if size not in self.SIZES:
            raise ValueError(f"size must be one of: {self.SIZES}")
        self.size = size
        self.model_name = f"sentence-transformers/sentence-t5-{size}"

    def namespace(self):
        return f"embed/sentence-t5/{self.size}"

    def shape(self):
        return self.SIZES[self.size]

    def __call__(self, text):
        return self.model.encode(text)


models = {
    'flag': EmbedFlag,
    'mpnet': EmbedMpnet,
    'minilm': EmbedMinilm,
    'gtr-t5': EmbedGtrT5,
    'sentence-t5': EmbedSentenceT5,
    'flag-llm-embedder': EmbedFlagLLMEmbedder,
    'flag-reranker': EmbedFlagReranker,
}

config = {
    'default': EmbedFlag,
}

def list_models():
    return list(models.keys())

def set_default_model(name):
    config['default'] = models[name]

def get_default_model():
    return config['default']

# Default
class Embed(EmbedBase):
    def __new__(cls, *args, **kwds):
        cls2 = get_default_model()
        self = cls2(*args, **kwds)
        return self

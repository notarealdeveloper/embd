from embd import Space
from embd import EmbedDefault
from mmry import CacheDefault

def test_think():
    space = Space(embd)
    embed = think('hello world')
    assert embed.ndim == 1
    assert len(embed) == EmbedDefault.SIZES['large']

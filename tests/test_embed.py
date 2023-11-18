from embd import Space
from embd import Embed

def test_think():
    space = Space()
    embed = think('hello world')
    assert embed.ndim == 1
    assert len(embed) == EmbedDefault.SIZES['large']

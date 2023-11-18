from embd import Space
from embd import Embed

def test_think():
    space = Space(Embed('large'))
    embed = space.think('hello world')
    assert embed.ndim == 1
    assert len(embed) == Embed.SIZES['large']

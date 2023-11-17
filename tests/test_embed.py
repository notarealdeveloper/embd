from mbd import EmbedDefault

def test_embed():
    model = EmbedDefault()
    embed = model('hello world')
    assert embed.ndim == 1

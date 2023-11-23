
def test_space():
    from embd import Space, EmbedFlag, EmbedMpnet

    space = Space(EmbedFlag('small'))
    embed = space.think('hello world')
    assert embed.ndim == 1
    assert len(embed) == EmbedFlag.SIZES['small']

    space = Space(EmbedFlag('large'))
    embed = space.think('hello world')
    assert embed.ndim == 1
    assert len(embed) == EmbedFlag.SIZES['large']

    space = Space(EmbedMpnet('base'))
    embed = space.think('hello world')
    assert embed.ndim == 1
    assert len(embed) == EmbedMpnet.SIZES['base']

def test_list():

    from embd import List
    things  = [
        'red cow',
        'orange dog',
        'yellow horse',
        'green chicken',
        'blue cat',
        'purple rabbit',
    ]
    colors  = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    animals = ['cow', 'dog', 'horse', 'chicken', 'cat', 'rabbit']
    Things  = List(things)
    Colors  = List(colors)
    Animals = List(animals)

    assert Things.top(Colors) == colors
    assert Things.top(Animals) == animals
    assert Things.top(colors) == colors
    assert Things.top(animals) == animals
    assert List(things[::-1]).top(colors) == colors[::-1]
    assert List(things[::-1]).top(animals) == animals[::-1]


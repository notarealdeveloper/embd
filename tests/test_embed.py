
def test_space():
    from embd import Space, Embed

    space = Space(Embed('small'))
    embed = space.think('hello world')
    assert embed.ndim == 1
    assert len(embed) == Embed.SIZES['small']

    space = Space(Embed('large'))
    embed = space.think('hello world')
    assert embed.ndim == 1
    assert len(embed) == Embed.SIZES['large']


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

    assert Things.get(Colors) == colors
    assert Things.get(Animals) == animals
    assert Things.get(colors) == colors
    assert Things.get(animals) == animals
    assert List(things[::-1]).get(colors) == colors[::-1]
    assert List(things[::-1]).get(animals) == animals[::-1]


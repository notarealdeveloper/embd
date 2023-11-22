#!/usr/bin/env python3

__all__ = [
    'Object',
    'List',
    'Dict',
    'promote',
]

import is_instance

# ======================

def promote(o):
    import numpy as np
    if isinstance(o, (list, tuple, set)):
        return List(o)
    elif isinstance(o, dict):
        return Dict(o)
    elif isinstance(o, str):
        return List([o])
    elif isinstance(o, bytes):
        return List([o.decode()])
    elif isinstance(o, (List, Dict)):
        return o
    elif isinstance(o, F):
        return o
    elif isinstance(o, np.ndarray) and o.ndim == 2:
        return List(o)
    elif isinstance(o, np.ndarray) and o.ndim == 1:
        return List([o])
    else:
        raise TypeError(o)

# ======================

def series(o, space=None):
    import pandas as pd
    from .space import Space
    if space is None:
        space = Space()
    if isinstance(o, pd.Series):
        return o
    if isinstance(o, str):
        return pd.Series(space.think(o), name=o)
    raise TypeError

def frame(o, space=None):
    import numpy as np
    import pandas as pd
    from .space import Space
    if space is None:
        space = Space()
    if isinstance(o, pd.DataFrame):
        return o
    if isinstance(o, pd.Series):
        return pd.DataFrame(o)
    if is_instance(o, str):
        o = [o]
    if is_instance(o, list[str]):
        return pd.DataFrame(space.think(o).T, columns=o)
    if is_instance(o, dict[str, str]):
        return pd.DataFrame(space.think(list(o.values())).T, columns=list(o.keys()))
    if is_instance(o, np.ndarray):
        return pd.DataFrame(o.T, columns=list(range(len(o))))
    if is_instance(o, list[np.ndarray]):
        return pd.DataFrame(np.stack(o).T, columns=list(range(len(o))))
    if is_instance(o, dict[int, np.ndarray]):
        return pd.DataFrame(np.stack(list(o.values())).T, columns=list(o.keys()))
    raise TypeError(o)

# ======================

class O:
    def __init__(self, o):
        self.o = o

class D(O):
    def __init__(self, o):
        self.d = o
        super().__init__(o)

    def __getitem__(self, i):
        return self.d[i]

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def __repr__(self):
        return repr(self.d)

    def keys(self):
        return list(self.d.keys())

    def values(self):
        return list(self.d.values())

    def items(self):
        return list(self.d.items())

class L(O):
    def __init__(self, o):
        self.l = o
        super().__init__(o)

    def __getitem__(self, i):
        return self.l[i]

    def __len__(self):
        return len(self.l)

    def __iter__(self):
        return iter(self.l)

    def __add__(self, other):
        import itertools
        return L([''.join(p) for p in itertools.product(self, other)])

    def __repr__(self):
        return repr(self.l)

class F(O):
    def __init__(self, o):
        self.f = frame(o)
        super().__init__(o)

    def names(self):
        return self.f.columns

    def centerofmass(self):
        # center of mass: different from subtracting mean from self!
        return self.f.mean(axis=1)

    def dot(self, other):
        return self.f.T @ other.f

    def sims(self, other):
        return self.dot(self.at(self), other.at(other))

    def at(self, other):
        return F(self.f.subtract(other.centerofmass(), axis=0))

    def centered(self):
        return F(self.at(self))

    def centered_at(self, other):
        return self.at(other)

    def __matmul__(self, other):
        return self.dot(other)

    def __call__(self, other=None):
        if other is None: other = self
        return self.at(other)

    def __repr__(self):
        return repr(self.f)

    def __getitem__(self, i):
        return self.f[i]

    def __eq__(self, other):
        import numpy as np
        return np.allclose(self.f, other.f)

    def __sub__(self, other):
        return self.f - other.f

    def get(self, other):
        import numpy as np
        s = self @ promote(other)
        i = np.argsort(-s)
        s.iloc[:, :] = s.columns.values[i]
        s.columns = list(range(1, len(s.columns)+1))
        return s

    def top(self, other, n=1):
        s = self.get(other)
        return s[1].tolist()

class S(O):
    def __init__(self, o):
        self.s = series(o)
        super().__init__(o)

    def name(self):
        return self.s.name

    def __sub__(self, other):
        return self.s - other.s

    def __repr__(self):
        return repr(self.s)


class Object(S):

    def __init__(self, o):
        super().__init__(o)

    def __repr__(self):
        return repr(self.s)

class List(F, L):

    def __init__(self, o):
        super().__init__(list(o))

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.l[i]
        elif is_instance(i, str | list[str]):
            return self.f[i]
        else:
            raise IndexError(i)

    def __iter__(self):
        return iter(self.o)

class Dict(F, D):

    def __init__(self, o):
        super().__init__(dict(o))

    def __getitem__(self, i):
        if is_instance(i, str | list[str]):
            return self.f[i]
        else:
            raise IndexError(i)


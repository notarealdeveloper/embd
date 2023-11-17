#!/usr/bin/env python3

__all__ = [
    'grep',
    'greps',
    'format_grep',
    'promote',
    'Object',
    'List',
    'Dict',
]

import assure
import itertools
import is_instance
import numpy as np
import pandas as pd

import mbd

def embed(o):
    return mbd.think(o)

def series(o):
    if isinstance(o, pd.Series):
        return o
    if isinstance(o, str):
        return pd.Series(embed(o), name=o)
    raise TypeError

def frame(o):
    if isinstance(o, pd.DataFrame):
        return o
    if isinstance(o, pd.Series):
        return pd.DataFrame(o)
    if is_instance(o, str):
        o = [o]
    if is_instance(o, list[str]):
        return pd.DataFrame(embed(o).T, columns=o)
    if is_instance(o, dict[str, str]):
        return pd.DataFrame(embed(list(o.values())).T, columns=list(o.keys()))
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
        return np.allclose(self.f, other.f)

    def __sub__(self, other):
        return self.f - other.f

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


def argsort_reverse(a):
    return np.argsort(-a, axis=-1)


SIMILARITY_DEFINITIONS = [
    'a @ b',
    'a @ b(a)',
    'a @ b(b)',
    'a(a) @ b',
    'a(a) @ b(a)',
    'a(a) @ b(b)',
    'a(b) @ b',
    'a(b) @ b(a)',
    'a(b) @ b(b)',
]

def promote(o):
    if isinstance(o, (list, tuple, set)):
        return List(o)
    elif isinstance(o, dict):
        return Dict(o)
    elif isinstance(o, str):
        return Object(o)
    elif isinstance(o, bytes):
        return Object(o.decode())
    elif isinstance(o, (List, Dict, Object)):
        return o
    else:
        raise TypeError(o)

def grep(queries, keys, similarity_definition='a @ b', *, n=None):
    q = promote(assure.plural(queries))
    k = promote(assure.plural(keys))
    s = (lambda a,b: eval(similarity_definition))(q,k)
    i = argsort_reverse(s)
    g = 0*s
    g.iloc[:, :] = s.columns.values[i]
    g.columns = list(range(1, len(g.columns)+1))
    g.columns.name = similarity_definition
    g = g.iloc[:, 0:n]
    return g

def greps(queries, keys, *, n=None):
    """ Evaluate possible definitions of grep interactively
        by varying the definition of similarity """
    gs = []
    for similarity_definition in SIMILARITY_DEFINITIONS:
        g = grep(queries, keys, similarity_definition, n=n)
        gs.append(g)
    return gs

def format_grep(g, n=None):
    """ format and prepare a grep for stdout """
    if n is None:
        n = 1
    g = g.iloc[:, 0:n]
    q = g.index.tolist()
    k = g.to_csv(index=False, header=False)
    d = pd.DataFrame(q, index=k.splitlines())
    o = d.to_csv(index=True,header=False,sep=':').strip()
    return o


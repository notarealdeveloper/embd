from embd.bin import think


def test_iter_files_expands_dirs_and_files_in_order(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "b.txt").write_text("b")
    (root / "a.txt").write_text("a")
    child = root / "child"
    child.mkdir()
    (child / "c.txt").write_text("c")
    file = tmp_path / "file.txt"
    file.write_text("file")

    assert list(think.iter_files([str(root), str(file)])) == [
        str(root / "a.txt"),
        str(root / "b.txt"),
        str(child / "c.txt"),
        str(file),
    ]


def test_embed_files_uses_one_space_per_worker_not_per_file(monkeypatch, tmp_path):
    files = []
    for n in range(3):
        file = tmp_path / f"{n}.txt"
        file.write_text(str(n))
        files.append(str(file))

    spaces = []
    embedded = []

    class FakeSpace:
        def __init__(self):
            spaces.append(self)

        def get_from_path(self, path):
            embedded.append(path)

    class FakePool:
        def __init__(self, processes, initializer):
            self.processes = processes
            self.initializer = initializer

        def __enter__(self):
            self.initializer()
            return self

        def __exit__(self, *args):
            pass

        def imap_unordered(self, function, paths):
            for path in paths:
                yield function(path)

    pools = []

    def pool(processes, initializer):
        pool = FakePool(processes, initializer)
        pools.append(pool)
        return pool

    monkeypatch.setattr(think.embd, "Space", FakeSpace)
    monkeypatch.setattr(think.multiprocessing, "Pool", pool)

    assert sorted(think.embed_files(files, processes=2)) == sorted(files)
    assert len(spaces) == 1
    assert pools[0].processes == 2
    assert sorted(embedded) == sorted(files)

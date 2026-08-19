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


def test_filter_embeddable_files_accepts_text_and_csv(monkeypatch, tmp_path):
    text = tmp_path / "text"
    csv = tmp_path / "csv"
    text.write_text("hello")
    csv.write_text("a,b\n1,2\n")

    file_types = {
        str(text): ("text/plain", "unicode text, utf-8 text"),
        str(csv): ("text/csv", "csv ascii text"),
    }

    monkeypatch.setattr(think, "file_type", lambda path: file_types[path])

    assert think.filter_embeddable_files([str(text), str(csv)]) == [
        str(text),
        str(csv),
    ]


def test_filter_embeddable_files_skips_media_and_large_data(monkeypatch, tmp_path):
    jpg = tmp_path / "image.jpg"
    mp4 = tmp_path / "movie.mp4"
    data = tmp_path / "data"
    jpg.write_bytes(b"jpg")
    mp4.write_bytes(b"mp4")
    data.write_bytes(b"x" * (think.MAX_DATA_BYTES + 1))

    file_types = {
        str(jpg): ("image/jpeg", "jpeg image data"),
        str(mp4): ("video/mp4", "iso media, mp4"),
        str(data): ("application/octet-stream", "data"),
    }

    monkeypatch.setattr(think, "file_type", lambda path: file_types[path])

    assert think.filter_embeddable_files([str(jpg), str(mp4), str(data)]) == []


def test_filter_embeddable_files_allows_only_a_few_small_weird_files(monkeypatch, tmp_path):
    files = []
    for n in range(5):
        file = tmp_path / f"weird-{n}"
        file.write_bytes(b"\x80")
        files.append(str(file))

    monkeypatch.setattr(
        think,
        "file_type",
        lambda path: ("application/octet-stream", "data"),
    )

    assert think.filter_embeddable_files(files) == files[:3]


def test_embed_files_force_bypasses_file_filter(monkeypatch, tmp_path):
    files = []
    for name in ("text", "image.jpg"):
        file = tmp_path / name
        file.write_text(name)
        files.append(str(file))

    embedded = []

    class FakeSpace:
        def get_from_path(self, path):
            embedded.append(path)

    class FakePool:
        def __init__(self, processes, initializer):
            self.initializer = initializer

        def __enter__(self):
            self.initializer()
            return self

        def __exit__(self, *args):
            pass

        def imap_unordered(self, function, paths):
            for path in paths:
                yield function(path)

    monkeypatch.setattr(think.embd, "Space", FakeSpace)
    monkeypatch.setattr(think.multiprocessing, "Pool", FakePool)
    monkeypatch.setattr(think, "filter_embeddable_files", lambda paths: [])

    assert sorted(think.embed_files(files, force=True)) == sorted(files)
    assert sorted(embedded) == sorted(files)

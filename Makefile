PKG = embd
PYTHON ?= python
PIP ?= pip

build: build-deps
	$(PYTHON) -m build

install:
	$(PYTHON) -m $(PIP) install dist/*.tar.gz

develop:
	$(PYTHON) -m $(PIP) install -e .

check:
	pytest -v tests

uninstall:
	$(PYTHON) -m $(PIP) uninstall $(PKG)

clean:
	rm -rvf dist/ build/ src/*.egg-info

push-test:
	$(PYTHON) -m $(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi dist/*

pull-test:
	$(PYTHON) -m $(PIP) install -i https://test.pypi.org/simple/ $(PKG)

push-prod:
	$(PYTHON) -m $(PIP) install twine
	$(PYTHON) -m twine upload dist/*

pull-prod:
	$(PYTHON) -m $(PIP) install $(PKG)

build-deps:
	@$(PYTHON) -c 'import build' &>/dev/null || $(PYTHON) -m $(PIP) install build

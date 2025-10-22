PYTHON ?= python
SPHINXBUILD = pixi run -e doc sphinx-build
SOURCEDIR = .
BUILDDIR = _build

.PHONY: html clean serve apidoc

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf $(BUILDDIR)/*

serve:
	python -m http.server 8000 --bind 127.0.0.1 --directory $(BUILDDIR)/html

apidoc:
	pixi run -e doc sphinx-apidoc -o docs/_autosummary numberlink

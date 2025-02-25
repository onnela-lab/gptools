.PHONY : all docs doctests

all : doctests

docs :
	rm -rf docs/_build
	sphinx-build -n -W . docs/_build

doctests :
	sphinx-build -n -W -b doctest . docs/_build

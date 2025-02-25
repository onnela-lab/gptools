.PHONY : all clean docs doctests

all : doctests

docs :
	rm -rf docs/_build
	sphinx-build -n -W . docs/_build

doctests :
	sphinx-build -n -W -b doctest . docs/_build

clean :
	rm -rf docs/_build docs/jupyter_execute docs/.jupyter_cache

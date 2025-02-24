MODULES = stan util
# Targets for the packages.
DISTS = ${MODULES:%=%/dist}
DOCS = ${MODULES:%=%/docs}
DOCTESTS = ${MODULES:%=%/doctests}
LINT = ${MODULES:%=%/lint}
TESTS = ${MODULES:%=%/tests}

all : docs doctests lint tests dists

.PHONY : ${DISTS} ${DOCS} ${DOCTESTS} ${LINT} ${TESTS}

docs : ${DOCS}
${DOCS} : %/docs :
	PROJECT=$* sphinx-build -n -W . $@/_build

doctests : ${DOCTESTS}
${DOCTESTS} : %/doctests :
	PROJECT=$* sphinx-build -n -W -b doctest . $*/docs/_build

lint : ${LINT}
${LINT} : %/lint :
	flake8 $*
	black --check $*

tests : ${TESTS}
${TESTS} : %/tests :
	pytest -v --cov=gptools.$* --cov-report=term-missing --cov-fail-under=100 $@

dists : ${DISTS}
${DISTS} : %/dist :
	cd $* && python setup.py sdist
	twine check $@/*.tar.gz

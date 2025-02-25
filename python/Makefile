MODULES = stan util
# Targets for the packages.
DISTS = ${MODULES:%=%/dist}
LINT = ${MODULES:%=%/lint}
TESTS = ${MODULES:%=%/tests}
.PHONY : ${DISTS} ${LINT} ${TESTS}

all : all lint tests dists

lint : ${LINT}
${LINT} : %/lint :
	flake8 $*
	black --check $*

dists : ${DISTS}
${DISTS} : %/dist :
	cd $* && python setup.py sdist
	twine check $@/*.tar.gz

tests : ${TESTS}
${TESTS} : %/tests :
	pytest -v --cov=gptools.$* --cov-report=term-missing --cov-fail-under=100 $@

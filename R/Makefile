.PHONY : build clean submit

build : clean
	Rscript -e 'devtools::document()'
	rm -f *.tar.gz
	NOT_CRAN=true R CMD build .
	R CMD check --as-cran *.tar.gz

clean :
	rm -rf *.Rcheck *.tar.gz

submit : clean
	echo "Start submission by typing 'devtools::submit_cran()'"
	NOT_CRAN=true R

.PHONY : build check stan_files man vignettes
SOURCE_DIR = ../stan/gptools/stan/gptools
SOURCE_FILES = $(wildcard $(SOURCE_DIR)/*.stan)
TARGET_DIR = inst/extdata
TARGET_FILES = $(foreach f,$(SOURCE_FILES),$(TARGET_DIR)/gptools/$(notdir $f))

# Targets to copy over the files whenever they need updating.
stan_files : $(TARGET_FILES) vignettes/getting_started.stan

$(TARGET_FILES) : $(TARGET_DIR)/gptools/%.stan : $(SOURCE_DIR)/%.stan
	mkdir -p $(dir $@)
	cp $< $@

vignettes/getting_started.stan : ../stan/docs/getting_started/getting_started.stan
	mkdir -p $(dir $@)
	cp $< $@

# Build the vignette.
vignettes/getting_started.html : vignettes/getting_started.Rmd vignettes/getting_started.stan stan_files
	Rscript -e "devtools::build_rmd('$<')"

man :
	Rscript -e 'devtools::document()'


build :
	mkdir -p build
	cd build && R CMD build ..

check : build
	mkdir -p check
	cd check && R CMD check ../build/*.tar.gz

#' Get the 'gptoolsStan' include path for compiling 'Stan' programs.
#'
#' @export
#' @return Path to the directory containing source files for 'gptoolsStan' as characters.
#' @examples
#' \dontrun{
#' library(cmdstanr)
#' library(gptoolsStan)
#'
#' # Compile the model with paths set up to include 'Stan' sources from 'gptoolsStan'.
#' model <- cmdstan_model(
#'     stan_file = "/path/to/your/model.stan",
#'     include_paths = gptools_include_path(),
#' )
#' }
gptools_include_path <- function() {
    version <- package_version(cmdstanr::cmdstan_version())
    if (version < package_version("2.36.0")) {
        warning(
            sprintf(
                paste(
                    "cmdstan<2.36.0 had a bug in the evaluation of Matern 3/2 kernels",
                    "with different length scales for different dimensions; see",
                    "https://github.com/stan-dev/math/pull/3084 for details. Your",
                    "model may yield unexpected results or crash if you use",
                    "nearest-neighbor Gaussian processes with Matern 3/2 kernels.",
                    "Your cmdstan version is %s; consider upgrading."
                ),
                version
            )
        )
    }
    system.file("extdata", package = "gptoolsStan")
}

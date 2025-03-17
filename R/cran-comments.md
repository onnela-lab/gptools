# Submission of version 1.0.0.

The DOI in the CITATION is for a new JSS publication that will be registered after publication on CRAN.

# Submission of version 0.2.0.

This submission adds functionality to specify different length scales in different dimensions for Gaussian processes on graphs.


# Revisions in response to comments from 2023-12-18 for resubmission of 0.1.0.

Thank you for the comments. Please find point-by-point responses below. Code changes are available at https://github.com/onnela-lab/gptoolsStan/pull/4.

> Please always write package names, software names and API (application programming interface) names in single quotes in title and description. e.g: --> 'Stan' Please note that package names are case sensitive.

We have wrapped 'Stan' and 'cmdstanr' in single quotes.

> Please always explain all acronyms in the description text. -> 'GP'

We have expanded acronyms.

> If there are references describing the methods in your package, please add these in the description field of your DESCRIPTION file in the form
> authors (year) <doi:...>
> authors (year) <arXiv:...>
> authors (year, ISBN:...)
> or if those are not available: <https:...>
> with no space after 'doi:', 'arXiv:', 'https:' and angle brackets for auto-linking. (If you want to add a title as well please put it in quotes: "Title")

We have added a reference.

> Please add `\value` to `.Rd` files regarding exported methods and explain the functions results in the documentation. Please write about the structure of the output (class) and also what the output means. (If a function does not return a value, please document that too, e.g. `\value{No return value, called for side effects}` or similar)
> Missing Rd-tags:
>
>    - `gptools_include_path.Rd: \value`

We have added a `\value` tag.

> Please add small executable examples in your Rd-files to illustrate the use of the exported function but also enable automatic testing.

We have added an example to the `.Rd` files. It is however wrapped in `\dontrun` because running the example requires that 'cmdstan' and 'cmdstanr' are installed.

# R CMD check results for first submission 0.1.0.

0 errors | 0 warnings | 1 note

* This is a new release.

# This script converts from RDA to CSV.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    stop("must supply exactly two arguments", call. = FALSE)
}
load(args[1])
frame <- get(args[2])
write.csv(frame, file = args[3], row.names = FALSE)
print(nrow(frame))

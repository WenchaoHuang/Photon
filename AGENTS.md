# Photon Development Guidelines

1. Follow `.gitattributes` for line endings. Repository text files use LF by
   default; preserve explicit exceptions such as `.bat` and `.cmd` files. Use
   UTF-8 without BOM and tabs for indentation. Do not alter the encoding,
   indentation, or line endings of untouched lines.

2. Keep API and CUDA behavior consistent with adjacent code. Reuse existing
   Photon types, `PHOTON_*` macros, logging, and error-handling patterns instead of
   creating parallel helpers or changing error semantics without a clear need.

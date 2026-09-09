"""Warning filters applied before any heavy import.

Two warnings show up on every command and neither says anything about this
project:

* ``NotOpenSSLWarning`` from urllib3, because the interpreter's ``ssl`` module
  is linked against LibreSSL rather than OpenSSL. That is how macOS ships
  Python; it does not affect anything here, and the fix belongs to the
  interpreter, not the repository.
* ``RuntimeWarning: ... encountered in matmul`` from numpy, which reports a
  floating-point flag the transformer's forward pass left set in BLAS against
  the next matmul it evaluates. Similarity values are checked for finiteness
  where they are computed.

Import this module before sentence-transformers or urllib3.
"""

import warnings


def apply() -> None:
    warnings.filterwarnings(
        "ignore", message=".*OpenSSL.*", module="urllib3"
    )
    warnings.filterwarnings(
        "ignore", message=".*encountered in matmul", category=RuntimeWarning
    )


apply()

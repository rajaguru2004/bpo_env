# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BPO Customer Support Resolution Environment."""

# Support both:
#   - Package import:  import bpo_env          (relative imports work)
#   - Direct import:   import __init__ as pkg  (absolute imports needed)
try:
    from .client import CustomerSupportEnv
    from .models import CustomerSupportAction, CustomerSupportObservation
except ImportError:
    from client import CustomerSupportEnv  # type: ignore[no-redef]
    from models import CustomerSupportAction, CustomerSupportObservation  # type: ignore[no-redef]

__all__ = [
    "CustomerSupportAction",
    "CustomerSupportObservation",
    "CustomerSupportEnv",
]

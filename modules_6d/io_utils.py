"""Compatibility re-exports for legacy modules_6d imports.

The shared I/O helpers live in ``utils.io_utils`` in this tree, but some
modules still import them through ``modules_6d.io_utils``.
"""

from utils.io_utils import *  # noqa: F401,F403




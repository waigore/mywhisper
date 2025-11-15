"""
Textual frontend entrypoint for mywhisper (myw).
"""

from typing import TYPE_CHECKING

__all__ = ["MywApp"]

if TYPE_CHECKING:
    from .app import MywApp as _MywApp


def __getattr__(name: str):
    if name == "MywApp":
        from .app import MywApp

        return MywApp
    raise AttributeError(name)
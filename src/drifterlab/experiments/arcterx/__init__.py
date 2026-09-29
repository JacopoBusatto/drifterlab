"""ARCTERX MicroSVP adapter. Campaign timestamps are interpreted as UTC."""

from .microsvp import read_microsvp

__all__ = ["read_microsvp"]

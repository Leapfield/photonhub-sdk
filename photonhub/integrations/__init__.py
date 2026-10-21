"""Bridges from layout tools into PhotonHub.

Each bridge is a submodule behind its own optional extra, imported by name so
the base SDK never loads a layout package it does not need:

>>> from photonhub.integrations import gdsfactory as phgf   # photonhub[gdsfactory]
"""

__all__ = []

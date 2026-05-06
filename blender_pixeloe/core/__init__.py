"""Pure-numpy port of the PixelOE algorithm.

This subpackage must remain importable outside Blender. No `import bpy` here
or in any submodule.
"""

from .pixelize import pixelize

__all__ = ["pixelize"]

"""Blender addon: PixelOE pixelization.

Algorithm by Shih-Ying Yeh (KohakuBlueleaf), upstream:
https://github.com/KohakuBlueleaf/PixelOE
"""

bl_info = {
    "name": "PixelOE",
    "author": "Simon Sorkin",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "Image Editor > Sidebar > PixelOE",
    "description": "Pixelize images with the PixelOE algorithm",
    "category": "Image",
}


def register():
    from . import operators, panels
    operators.register()
    panels.register()


def unregister():
    from . import operators, panels
    panels.unregister()
    operators.unregister()

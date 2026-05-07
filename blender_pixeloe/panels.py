"""Image Editor N-panel for the PixelOE addon."""
from __future__ import annotations

import bpy

from .operators import PIXELOE_OT_pixelize_image


class PIXELOE_PT_panel(bpy.types.Panel):
    bl_label = "PixelOE"
    bl_idname = "PIXELOE_PT_panel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "PixelOE"

    def draw(self, context):
        layout = self.layout
        space = context.space_data
        settings = context.scene.pixeloe

        if space and space.image is not None:
            layout.label(text=space.image.name, icon='IMAGE_DATA')
        else:
            layout.label(text="No active image", icon='ERROR')

        col = layout.column(align=True)
        col.prop(settings, "mode")
        col.prop(settings, "target_size")
        col.prop(settings, "patch_size")
        col.prop(settings, "thickness")
        col.prop(settings, "colors")

        layout.prop(settings, "create_new")
        layout.operator(PIXELOE_OT_pixelize_image.bl_idname, icon='IMAGE')


_classes = (PIXELOE_PT_panel,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)

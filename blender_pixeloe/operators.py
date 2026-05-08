"""PixelOE operator and the Scene PropertyGroup that backs the N-panel."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import PurePath

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty

from .core.pixelize import pixelize
from .image_io import array_to_image, image_to_array

_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr",
    ".webp", ".bmp", ".tga", ".hdr",
}


def _strip_image_ext(name: str) -> str:
    """Drop a trailing image-format extension. Leaves Blender's `.001` auto-
    numbering alone."""
    suffix = PurePath(name).suffix.lower()
    if suffix in _IMAGE_EXTS:
        return PurePath(name).stem
    return name


class PixeloeSettings(bpy.types.PropertyGroup):
    target_size: IntProperty(
        name="Target Size",
        description="Pixel grid resolution. Output is roughly target_size * patch_size on the long edge",
        default=128, min=8, soft_max=512, max=2048,
    )
    patch_size: IntProperty(
        name="Patch Size",
        description="Block size for outline analysis and the size of each output pixel after upscale",
        default=16, min=1, soft_max=32, max=64,
    )
    upscale: BoolProperty(
        name="Upscale",
        description="Scale the result up by Patch Size with nearest-neighbour so each pixel becomes a chunky block. Off (default) keeps output at the pixel-grid resolution",
        default=False,
    )
    thickness: IntProperty(
        name="Outline Thickness",
        description="Erode/dilate iteration count for outline expansion. 0 disables outline pre-processing",
        default=2, min=0, max=8,
    )
    mode: EnumProperty(
        name="Mode",
        description="Downscale algorithm",
        items=[
            ('contrast', "Contrast", "Contrast-aware downscale (PixelOE default)"),
            ('k-centroid', "K-Centroid", "K-centroid per-tile quantization"),
        ],
        default='contrast',
    )
    colors: IntProperty(
        name="Colors",
        description="Palette size for color quantization. 0 disables quantization",
        default=0, min=0, max=256,
    )
    create_new: BoolProperty(
        name="Create New",
        description="Always create a new image datablock; otherwise overwrite the previous output",
        default=False,
    )


class PIXELOE_OT_pixelize_image(bpy.types.Operator):
    """Pixelize the active Image Editor image with the PixelOE algorithm.

    Runs synchronously and blocks the UI for a few seconds on large inputs;
    a wait cursor is shown for the duration"""
    bl_idname = "pixeloe.pixelize_image"
    bl_label = "Pixelize"
    bl_description = (
        "Pixelize the active image (PixelOE). Runs synchronously; the UI "
        "freezes with a wait cursor for a few seconds on large inputs"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (
            space is not None
            and getattr(space, 'type', None) == 'IMAGE_EDITOR'
            and space.image is not None
        )

    def execute(self, context):
        space = context.space_data
        src = space.image
        if src is None:
            self.report({'ERROR'}, "No active image in the Image Editor.")
            return {'CANCELLED'}

        settings = context.scene.pixeloe

        t_start = time.perf_counter()
        src_w, src_h = src.size
        print(
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
            f"PixelOE: pixelizing {src.name} ({src_w}x{src_h})",
            flush=True,
        )

        try:
            rgb = image_to_array(src)
        except (ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        wm = context.window_manager
        window = context.window
        wm.progress_begin(0, 1)
        window.cursor_set('WAIT')
        # Force one redraw cycle so the cursor change actually flushes to the
        # OS before we block the event loop. Without this, cursor_set is a
        # no-op on the Linux compositors I've tested.
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        try:
            try:
                out_rgb = pixelize(
                    rgb,
                    mode=settings.mode,
                    target_size=settings.target_size,
                    patch_size=settings.patch_size,
                    thickness=settings.thickness,
                    colors=settings.colors,
                    no_upscale=not settings.upscale,
                )
            except Exception as exc:
                self.report({'ERROR'}, f"PixelOE failed: {exc}")
                return {'CANCELLED'}
        finally:
            wm.progress_end()
            window.cursor_set('DEFAULT')

        out_name = f"{_strip_image_ext(src.name)}_pixel"
        out_image = array_to_image(
            out_rgb, out_name, overwrite=not settings.create_new
        )

        h, w = out_rgb.shape[:2]
        elapsed = time.perf_counter() - t_start
        print(
            f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
            f"PixelOE: done in {elapsed:.3f}s",
            flush=True,
        )
        self.report(
            {'INFO'},
            f"Pixelized {src.name} -> {out_image.name} ({w}x{h})",
        )
        return {'FINISHED'}


_classes = (PixeloeSettings, PIXELOE_OT_pixelize_image)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pixeloe = PointerProperty(type=PixeloeSettings)


def unregister():
    del bpy.types.Scene.pixeloe
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)

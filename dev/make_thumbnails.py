"""Build the preview thumbnails that the notebook and the banner embed from assets/.

    python dev/serve_views.py &
    python dev/shoot_views.py
    python dev/make_thumbnails.py

Reads the screenshots written by shoot_views.py, trims the surrounding whitespace, scales
them to a fixed width, and writes them to assets/. They are referenced by relative path, so
they have to be committed alongside the notebook.

Thumbnails are written at twice their display width so they stay sharp on a high density
display, and quantized to a palette because these are flat line plots and screenshots of UI
chrome, where a full 24 bit image costs several times the size for no visible gain.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

SOURCE = Path("/tmp")
ASSETS = Path(__file__).resolve().parent.parent / "assets"

DISPLAY_WIDTH = 210
SCALE = 2

# The view each section previews, and whether to trim whitespace on all sides. The served
# app is a full page whose left aligned sidebar is part of the point, so it is only trimmed
# at the bottom. servable_app is embedded in the notebook and the other three feed banner.png
# via make_banner.py. The commented entries are referenced by neither.
THUMBNAILS = {
    # "fast_hvplot_view": True,
    "image_view": True,
    # "image_levels_view": True,
    "traces_minimap_view": True,
    # "pyramid_app": True,
    "servable_app": False,
    "spike_raster_view": True,
    # "spike_field_view": True,
}


def content_box(image: Image.Image, all_sides: bool) -> tuple[int, int, int, int]:
    """Bounding box of everything that is not the near-white page background."""
    grey = image.convert("L")
    mask = grey.point(lambda value: 255 if value < 247 else 0)
    box = mask.getbbox()
    if box is None:
        return (0, 0, image.width, image.height)

    pad = 6
    left, top, right, bottom = box
    if not all_sides:
        left, top = 0, 0
    return (
        max(left - pad, 0),
        max(top - pad, 0),
        min(right + pad, image.width),
        min(bottom + pad, image.height),
    )


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    target_width = DISPLAY_WIDTH * SCALE
    total = 0

    for name, all_sides in THUMBNAILS.items():
        source = SOURCE / f"lav_{name}.png"
        if not source.exists():
            print(f"missing {source}, run shoot_views.py first")
            return 1

        image = Image.open(source).convert("RGB")
        image = image.crop(content_box(image, all_sides))
        height = round(image.height * target_width / image.width)
        image = image.resize((target_width, height), Image.LANCZOS)

        destination = ASSETS / f"{name}.png"
        image.convert("P", palette=Image.ADAPTIVE, colors=128).save(
            destination, optimize=True
        )
        size_kb = destination.stat().st_size / 1e3
        total += size_kb
        print(f"{destination.name:28} {target_width}x{height}  {size_kb:6.1f} KB")

    print(f"{len(THUMBNAILS)} thumbnails, {total / 1e3:.2f} MB total")
    return 0


if __name__ == "__main__":
    sys.exit(main())

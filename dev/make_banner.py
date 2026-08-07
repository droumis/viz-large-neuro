"""Generate the README banner image.

Layout: compact escalation tiers on the left, three real screenshots
on the right with data labels and strategy tags.

One color per strategy, used consistently everywhere:
  - teal (#0f766e): rasterize
  - amber (#b45309): downsample
  - purple (#7c3aed): pyramid / precomputed levels

Run: python dev/make_banner.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path("assets")
OUT = ASSETS / "banner.png"

WIDTH = 1100
PAD = 24
GAP = 14

# One color per strategy — used in tier accent bars, tier pills, and screenshot tags
COLOR_RASTERIZE = "#0f766e"
BG_RASTERIZE = "#ccfbf1"
COLOR_DOWNSAMPLE = "#b45309"
BG_DOWNSAMPLE = "#fef3c7"
COLOR_PYRAMID = "#7c3aed"
BG_PYRAMID = "#ede9fe"

# Neutral
BG = "#ffffff"
TEXT = "#1f2328"
SECONDARY = "#6b7280"
BORDER = "#d6dbdf"
PANEL_BG = "#f9fafb"


def get_font(size, bold=False, italic=False):
    if bold:
        names = ["/System/Library/Fonts/SFPro-Bold.otf",
                 "/System/Library/Fonts/Helvetica.ttc"]
    elif italic:
        names = ["/System/Library/Fonts/SFPro-RegularItalic.otf",
                 "/System/Library/Fonts/SFPro-Regular.otf",
                 "/System/Library/Fonts/Helvetica.ttc"]
    else:
        names = ["/System/Library/Fonts/SFPro-Regular.otf",
                 "/System/Library/Fonts/Helvetica.ttc"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def rounded_rect(draw, xy, fill=None, outline=None, width=1, radius=5):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_tag(draw, x, y, label, color, bg, font):
    tw = int(draw.textlength(label, font=font))
    w = tw + 14
    h = 20
    rounded_rect(draw, (x, y, x + w, y + h), fill=bg, outline=color, radius=3)
    draw.text((x + 7, y + 3), label, fill=color, font=font)
    return w + 6


def load_and_fit(name, target_w, target_h):
    img = Image.open(ASSETS / name)
    scale = target_w / img.width
    new_h = int(img.height * scale)
    img = img.resize((target_w, new_h), Image.LANCZOS)
    if new_h > target_h:
        top = (new_h - target_h) // 2
        img = img.crop((0, top, target_w, top + target_h))
    elif new_h < target_h:
        padded = Image.new("RGB", (target_w, target_h), BG)
        padded.paste(img, (0, (target_h - new_h) // 2))
        img = padded
    return img


def main():
    font_heading = get_font(13, bold=True)
    font_label = get_font(12, bold=True)
    font_sublabel = get_font(10)
    font_tag = get_font(10, bold=True)
    font_tier_num = get_font(12, bold=True)
    font_tier_title = get_font(12, bold=True)
    font_tier_tech = get_font(10, bold=True)
    font_note = get_font(10, italic=True)

    # Dimensions
    LEFT_W = 280
    DIVIDER_X = PAD + LEFT_W + PAD

    right_x = DIVIDER_X + 14
    avail_w = WIDTH - right_x - PAD
    col_w = (avail_w - 2 * GAP) // 3
    shot_h = 210

    LABEL_H = 36
    TAG_H = 30
    total_h = PAD + LABEL_H + shot_h + TAG_H + PAD

    # Load screenshots
    shots = [
        load_and_fit("image_view.png", col_w, shot_h),
        load_and_fit("traces_minimap_view.png", col_w, shot_h),
        load_and_fit("spike_raster_view.png", col_w, shot_h),
    ]

    labels = ["Fluorescence volume", "LFP, 35 channels", "9.3M spikes, 180 units"]
    sublabels = ["OME-Zarr over HTTP", "1250 Hz, Neuropixels NWB", "spikes and waveforms"]

    # Tags per screenshot — colors match the tiers on the left
    tags = [
        [("rasterize", COLOR_RASTERIZE, BG_RASTERIZE),
         ("pyramid", COLOR_PYRAMID, BG_PYRAMID)],
        [("downsample", COLOR_DOWNSAMPLE, BG_DOWNSAMPLE),
         ("rasterize", COLOR_RASTERIZE, BG_RASTERIZE),
         ("pyramid", COLOR_PYRAMID, BG_PYRAMID)],
        [("rasterize", COLOR_RASTERIZE, BG_RASTERIZE),
         ("precomputed bins", COLOR_PYRAMID, BG_PYRAMID)],
    ]

    canvas = Image.new("RGB", (WIDTH, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    # === LEFT COLUMN: Escalation tiers ===
    # Vertically center the left column within total_h
    tiers = [
        ("1", "Reduce what is sent", "rasterize", "downsample",
         COLOR_RASTERIZE, BG_RASTERIZE, COLOR_DOWNSAMPLE, BG_DOWNSAMPLE),
        ("2", "Reduce what is read", "multiscale pyramid", None,
         COLOR_PYRAMID, BG_PYRAMID, None, None),
        ("3", "Reduce what is stored", "reference, not display", None,
         SECONDARY, PANEL_BG, None, None),
    ]

    tier_h = 50
    tier_gap = 8
    n_tiers = len(tiers)
    left_content_h = (n_tiers * tier_h + (n_tiers - 1) * tier_gap
                      + 24)  # heading
    left_y_start = (total_h - left_content_h) // 2

    lx = PAD
    ly = left_y_start

    draw.text((lx, ly), "Composable Reductions",
              fill=TEXT, font=font_heading)
    ly += 24

    for i, (num, title, tech1, tech2, c1, bg1, c2, bg2) in enumerate(tiers):
        ty = ly + i * (tier_h + tier_gap)

        # Box
        rounded_rect(draw, (lx, ty, lx + LEFT_W, ty + tier_h),
                     fill=PANEL_BG, outline=BORDER, radius=4)

        # Left accent bar — use the primary technique color
        draw.rounded_rectangle((lx, ty, lx + 5, ty + tier_h),
                               radius=4, fill=c1)
        # Overwrite the right rounded corners of the accent with the box bg
        draw.rectangle((lx + 3, ty + 1, lx + 5, ty + tier_h - 1), fill=c1)

        # Number + title
        draw.text((lx + 14, ty + 8), f"{num}.", fill=SECONDARY, font=font_tier_num)
        draw.text((lx + 32, ty + 8), title, fill=TEXT, font=font_tier_title)

        # Technique pill(s)
        pill_x = lx + 32
        pill_y = ty + 28
        tw1 = int(draw.textlength(tech1, font=font_tier_tech))
        rounded_rect(draw, (pill_x, pill_y, pill_x + tw1 + 12, pill_y + 17),
                     fill=bg1, outline=c1, radius=3)
        draw.text((pill_x + 6, pill_y + 2), tech1, fill=c1, font=font_tier_tech)

        if tech2:
            pill_x2 = pill_x + tw1 + 18
            tw2 = int(draw.textlength(tech2, font=font_tier_tech))
            rounded_rect(draw, (pill_x2, pill_y, pill_x2 + tw2 + 12, pill_y + 17),
                         fill=bg2, outline=c2, radius=3)
            draw.text((pill_x2 + 6, pill_y + 2), tech2, fill=c2, font=font_tier_tech)



    # Vertical divider
    draw.line([(DIVIDER_X, PAD + 4), (DIVIDER_X, total_h - PAD - 4)],
              fill=BORDER, width=1)

    # === RIGHT: Screenshots ===
    for i in range(3):
        sx = right_x + i * (col_w + GAP)

        # Labels
        draw.text((sx + col_w // 2, PAD + 4), labels[i],
                  fill=TEXT, font=font_label, anchor="mt")
        draw.text((sx + col_w // 2, PAD + 20), sublabels[i],
                  fill=SECONDARY, font=font_sublabel, anchor="mt")

        # Screenshot with border
        sy = PAD + LABEL_H
        draw.rectangle((sx - 1, sy - 1, sx + col_w + 1, sy + shot_h + 1),
                       outline=BORDER)
        canvas.paste(shots[i], (sx, sy))

        # Tags centered below
        tag_widths = [int(draw.textlength(t, font=font_tag)) + 20
                      for t, _, _ in tags[i]]
        total_tw = sum(tag_widths) + 6 * (len(tags[i]) - 1)
        tx = sx + (col_w - total_tw) // 2
        ty = sy + shot_h + 6
        for tag_label, color, bg in tags[i]:
            consumed = draw_tag(draw, tx, ty, tag_label, color, bg, font_tag)
            tx += consumed

    canvas.save(OUT, optimize=True)
    print(f"wrote {OUT} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()

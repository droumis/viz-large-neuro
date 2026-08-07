"""Generate the three conceptual diagrams the notebook embeds from assets/.

    python dev/make_diagrams.py

One SVG per strategy. These are schematics, not plots of the real data, and they exist to
give a reader the shape of each idea before meeting an implementation of it.

SVG rather than PNG because these are line art, so they stay sharp at any size, cost a few
kilobytes, and diff as text. Written with the standard library only, so no drawing
dependency is added to the environment.

Where a diagram claims a relationship, the drawing computes it rather than approximating it
by eye. The aggregation grid in the rasterization diagram is the actual binned count of the
points drawn beside it, and the highlighted samples in the downsampling diagram are the
actual per-column minima and maxima of the curve drawn above them.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"

INK = "#1f2328"
MUTED = "#6b7280"
FAINT = "#d6dbdf"
ACCENT = "#0f766e"
ACCENT_SOFT = "#ccfbf1"
DROP = "#c2410c"
FONT = "-apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif"

def head(w, h, alt):
    return HEAD.format(w=w, h=h, wb=w - 1, hb=h - 1, muted=MUTED, border=BORDER, alt=alt)


HEAD = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" \
height="{h}" role="img" aria-label="{alt}">
<defs>
<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" \
orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{muted}"/></marker>
</defs>
<rect x="0.5" y="0.5" width="{wb}" height="{hb}" fill="#ffffff" stroke="{border}" stroke-width="1" rx="4"/>
"""


BORDER = "#b8c0c4"


def text(x, y, body, size=13, fill=INK, anchor="start", weight="normal", style="normal"):
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}">{body}</text>\n'
    )


def arrow(x1, y1, x2, y2, colour=MUTED):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
        f'stroke-width="1.6" marker-end="url(#a)"/>\n'
    )


def rect(x, y, w, h, fill="none", stroke=MUTED, width=1.2, radius=0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"{d}/>\n'
    )


def signal(n, seed=3):
    """A deterministic wiggle with structure at more than one scale."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = i / n
        value = (
            math.sin(t * 7.5 * math.pi) * 0.55
            + math.sin(t * 23 * math.pi) * 0.22
            + math.sin(t * 2.1 * math.pi) * 0.30
            + rng.uniform(-0.13, 0.13)
        )
        out.append(value)
    return out


# --- 1. rasterization ---------------------------------------------------------------


def rasterize_diagram() -> str:
    w, h = 760, 288
    cols, rows = 9, 7
    box = (24, 56, 168, 140)

    rng = random.Random(11)
    points = []
    for _ in range(520):
        # Two clumps plus a scatter, so the aggregated grid has something to show.
        pick = rng.random()
        if pick < 0.45:
            px, py = rng.gauss(0.34, 0.13), rng.gauss(0.40, 0.15)
        elif pick < 0.8:
            px, py = rng.gauss(0.68, 0.10), rng.gauss(0.68, 0.12)
        else:
            px, py = rng.random(), rng.random()
        if 0 <= px <= 1 and 0 <= py <= 1:
            points.append((px, py))

    counts = [[0] * cols for _ in range(rows)]
    for px, py in points:
        counts[min(int(py * rows), rows - 1)][min(int(px * cols), cols - 1)] += 1
    peak = max(max(row) for row in counts)

    svg = head(w, h, "A dense cloud of values aggregated into a coarse grid, and the grid sent to a browser")
    svg += text(24, 30, "Rasterization", size=15, weight="600")
    svg += text(24, 46, "for images, volumes, and dense clouds", size=12, fill=MUTED)

    bx, by, bw, bh = box
    svg += rect(bx, by, bw, bh, stroke=FAINT)
    for px, py in points:
        svg += (
            f'<circle cx="{bx + px * bw:.1f}" cy="{by + py * bh:.1f}" r="1.5" '
            f'fill="{INK}" opacity="0.55"/>\n'
        )
    svg += text(bx, by + bh + 18, "the array", size=12, weight="600")
    svg += text(bx, by + bh + 33, "more values than pixels", size=11, fill=MUTED)

    svg += arrow(bx + bw + 12, by + bh / 2, bx + bw + 68, by + bh / 2)
    svg += text(bx + bw + 40, by + bh / 2 - 12, "count", size=11, fill=MUTED, anchor="middle")
    svg += text(bx + bw + 40, by + bh / 2 + 22, "per cell", size=11, fill=MUTED, anchor="middle")

    gx, gy, gw, gh = 272, by, 168, bh
    cw, ch = gw / cols, gh / rows
    for r in range(rows):
        for c in range(cols):
            share = counts[r][c] / peak if peak else 0
            svg += (
                f'<rect x="{gx + c * cw:.1f}" y="{gy + r * ch:.1f}" width="{cw:.1f}" '
                f'height="{ch:.1f}" fill="{ACCENT}" fill-opacity="{share:.3f}" '
                f'stroke="{FAINT}" stroke-width="0.6"/>\n'
            )
    svg += rect(gx, gy, gw, gh, stroke=MUTED)
    svg += text(gx, gy + gh + 18, "one value per pixel", size=12, weight="600")
    svg += text(gx, gy + gh + 33, "computed on the server", size=11, fill=MUTED)

    svg += arrow(gx + gw + 12, gy + gh / 2, gx + gw + 68, gy + gh / 2)
    svg += text(gx + gw + 40, gy + gh / 2 - 12, "send", size=11, fill=MUTED, anchor="middle")

    sx, sy, sw, sh = 520, 44, 216, 164
    svg += rect(sx, sy, sw, sh, fill="#fbfcfc", stroke=MUTED, radius=6)
    svg += (
        f'<line x1="{sx}" y1="{sy + 20}" x2="{sx + sw}" y2="{sy + 20}" '
        f'stroke="{FAINT}" stroke-width="1.2"/>\n'
    )
    for i, cxp in enumerate((12, 24, 36)):
        svg += f'<circle cx="{sx + cxp}" cy="{sy + 10}" r="3" fill="{FAINT}"/>\n'
    ix, iy = sx + 22, sy + 36
    iw, ih = sw - 44, sh - 60
    icw, ich = iw / cols, ih / rows
    for r in range(rows):
        for c in range(cols):
            share = counts[r][c] / peak if peak else 0
            svg += (
                f'<rect x="{ix + c * icw:.1f}" y="{iy + r * ich:.1f}" width="{icw:.1f}" '
                f'height="{ich:.1f}" fill="{ACCENT}" fill-opacity="{share:.3f}"/>\n'
            )
    svg += rect(ix, iy, iw, ih, stroke=FAINT)
    svg += text(sx, sy + sh + 30, "the browser receives an image", size=12, weight="600")
    svg += text(sx, sy + sh + 45, "no individual value crosses the network", size=11, fill=MUTED)

    svg += text(
        24, h - 8,
        "The transfer is set by the size of the plot, so it does not grow with the array.",
        size=12, fill=INK, style="italic",
    )
    return svg + "</svg>\n"


# --- 2. downsampling ---------------------------------------------------------------


def downsample_diagram() -> str:
    w, h = 760, 340
    x0, x1 = 130, 720
    columns = 12
    per_column = 9
    n = columns * per_column

    values = signal(n)
    top_mid, top_amp = 112, 46
    bot_mid, bot_amp = 248, 46

    def px(i):
        return x0 + (x1 - x0) * i / (n - 1)

    svg = head(w, h, "Many samples per pixel reduced to the minimum and maximum at each pixel")
    svg += text(24, 30, "Downsampling", size=15, weight="600")
    svg += text(24, 46, "for long traces, where the line is the measurement", size=12, fill=MUTED)

    for c in range(columns + 1):
        x = x0 + (x1 - x0) * c / columns
        svg += (
            f'<line x1="{x:.1f}" y1="{top_mid - top_amp - 12}" x2="{x:.1f}" '
            f'y2="{top_mid + top_amp + 12}" stroke="{FAINT}" stroke-width="1"/>\n'
        )

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{px(i):.1f} {top_mid - v * top_amp:.1f}"
        for i, v in enumerate(values)
    )
    svg += f'<path d="{path}" fill="none" stroke="{MUTED}" stroke-width="1" opacity="0.75"/>\n'

    keep = []
    for c in range(columns):
        chunk = [(i, values[i]) for i in range(c * per_column, (c + 1) * per_column)]
        lo = min(chunk, key=lambda pair: pair[1])
        hi = max(chunk, key=lambda pair: pair[1])
        keep.extend(sorted({lo, hi}))

    kept = {i for i, _ in keep}
    for i, v in enumerate(values):
        if i not in kept:
            svg += (
                f'<circle cx="{px(i):.1f}" cy="{top_mid - v * top_amp:.1f}" r="2.6" '
                f'fill="#ffffff" stroke="{DROP}" stroke-width="1" opacity="0.75"/>\n'
            )
    for i, v in keep:
        svg += (
            f'<circle cx="{px(i):.1f}" cy="{top_mid - v * top_amp:.1f}" r="3" '
            f'fill="{ACCENT}"/>\n'
        )

    svg += text(24, top_mid - 26, "every sample", size=12, weight="600")
    svg += text(24, top_mid - 10, "one pixel of width", size=11, fill=MUTED)
    svg += text(24, top_mid + 4, "holds many", size=11, fill=MUTED)
    svg += text(24, top_mid + 26, "\u25cf kept", size=11, fill=ACCENT)
    svg += text(24, top_mid + 42, "\u25cb dropped", size=11, fill=DROP)

    svg += arrow((x0 + x1) / 2, top_mid + top_amp + 26, (x0 + x1) / 2, bot_mid - bot_amp - 30)

    kept_path = " ".join(
        f"{'M' if k == 0 else 'L'}{px(i):.1f} {bot_mid - v * bot_amp:.1f}"
        for k, (i, v) in enumerate(keep)
    )
    svg += f'<path d="{kept_path}" fill="none" stroke="{INK}" stroke-width="1.4"/>\n'
    for i, v in keep:
        svg += (
            f'<circle cx="{px(i):.1f}" cy="{bot_mid - v * bot_amp:.1f}" r="2.6" '
            f'fill="{ACCENT}"/>\n'
        )
    svg += text(24, bot_mid - 6, "what is drawn", size=12, weight="600")
    svg += text(24, bot_mid + 10, "real samples,", size=11, fill=MUTED)
    svg += text(24, bot_mid + 25, "same shape", size=11, fill=MUTED)

    svg += text(
        24, h - 10,
        "Keeping the extremes of each column preserves the envelope, so a brief spike "
        "survives the reduction.",
        size=12, fill=INK, style="italic",
    )
    return svg + "</svg>\n"


# --- 3. pyramid --------------------------------------------------------------------


def pyramid_diagram() -> str:
    w, h = 760, 404
    # Cell counts stand for resolution. Every level is written and every level spans the same
    # extent, which is the part most often misread as the coarse levels being shorter.
    levels = [
        (0, 64, "full resolution", True),
        (1, 32, "", True),
        (2, 16, "", True),
        (3, 8, "coarsest", True),
    ]
    bar_h, gap, bar_w = 22, 13, 236
    top = 150
    budget = 6  # stands for the width of the plot in pixels

    def panel(ox, title, subtitle, span):
        lo, hi = span
        counts = [int(cells * (hi - lo)) for _, cells, _, _ in levels]
        # The coarsest level that still supplies one sample per pixel of width.
        chosen = 0
        for index in reversed(range(len(levels))):
            if counts[index] >= budget:
                chosen = index
                break

        vx0, vx1 = ox + lo * bar_w, ox + hi * bar_w
        bottom = top + len(levels) * (bar_h + gap)

        out = text(ox, 80, title, size=12, weight="600")
        out += text(ox, 95, subtitle, size=11, fill=MUTED)

        # The viewport as a band through every level, since the zoom picks a range first.
        out += (
            f'<rect x="{vx0:.1f}" y="{top - 26}" width="{max(vx1 - vx0, 2):.1f}" '
            f'height="{bottom - top + 18}" fill="{ACCENT}" fill-opacity="0.07"/>\n'
        )
        out += rect(vx0, top - 26, max(vx1 - vx0, 2), 14, fill=ACCENT_SOFT,
                    stroke=ACCENT, width=1)
        out += text(min(vx0, ox + bar_w - 56), top - 32, "viewport", size=10, fill=ACCENT)

        for index, cells, _note, stored in levels:
            y = top + index * (bar_h + gap)
            selected = index == chosen
            cw = bar_w / cells
            for c in range(cells):
                inside = lo <= (c + 0.5) / cells <= hi
                fill = ACCENT if (selected and inside) else "#ffffff"
                out += (
                    f'<rect x="{ox + c * cw:.2f}" y="{y}" width="{cw:.2f}" height="{bar_h}" '
                    f'fill="{fill}" fill-opacity="{0.45 if selected and inside else 1}" '
                    f'stroke="{FAINT}" stroke-width="0.5"/>\n'
                )
            out += rect(ox, y, bar_w, bar_h,
                        stroke=ACCENT if selected else MUTED,
                        width=1.8 if selected else 1.0,
                        dash=None if stored else "4 3")
            colour = ACCENT if selected else MUTED
            weight = "600" if selected else "normal"
            out += text(ox + bar_w + 8, y + 15, f"{counts[index]}", size=11,
                        fill=colour, weight=weight)
            out += text(ox + bar_w + 30, y + 15, "in view", size=10, fill=colour)

        # The plot is the same number of pixels wide whatever the viewport covers, so the
        # strip is drawn at a fixed width and guide lines show the viewport stretched onto it.
        ry = bottom + 12
        out += (
            f'<path d="M{vx0:.1f} {bottom} L{ox} {ry} M{vx1:.1f} {bottom} '
            f'L{ox + bar_w} {ry}" stroke="{ACCENT}" stroke-width="0.8" '
            f'stroke-dasharray="3 3" opacity="0.55" fill="none"/>\n'
        )
        for c in range(budget):
            cwid = bar_w / budget
            out += rect(ox + c * cwid, ry, cwid, 11, fill="#ffffff", stroke=MUTED, width=0.7)
        out += text(ox, ry + 25, f"the plot, {budget} pixels wide", size=10, fill=MUTED)
        out += text(ox, ry + 46, f"level {chosen} is the coarsest level with", size=11,
                    fill=ACCENT, weight="600")
        out += text(ox, ry + 60, "at least one sample per pixel", size=11, fill=ACCENT)
        return out

    svg = head(w, h, "Four stored resolution levels covering the same extent, with the zoom selecting one")
    svg += text(24, 30, "Multiscale pyramid", size=15, weight="600")
    svg += text(24, 46, "for data larger than memory, or a reduction too slow to repeat",
                size=12, fill=MUTED)

    for index, _cells, note, _stored in levels:
        y = top + index * (bar_h + gap)
        svg += text(14, y + 10, f"level {index}", size=10, fill=MUTED)
        if note:
            svg += text(14, y + 21, note, size=9, fill=MUTED)

    svg += panel(104, "zoomed out", "the whole recording", (0.0, 1.0))
    svg += panel(436, "zoomed in far", "a short window", (0.44, 0.56))

    svg += (
        f'<line x1="415" y1="72" x2="415" y2="{top + 4 * (bar_h + gap) + 20}" '
        f'stroke="{FAINT}" stroke-width="1"/>\n'
    )
    svg += text(
        24, h - 10,
        "Every level covers the whole recording, and only the resolution differs. The finest "
        "is dashed because it is referenced, not written.",
        size=12, fill=INK, style="italic",
    )
    return svg + "</svg>\n"


# --- 4. the escalation ladder --------------------------------------------------------


TIERS = [
    (
        "Reduce what is sent",
        "rasterize, or downsample",
        "Aggregate to pixels, or keep a few real samples.",
        "Often enough on its own, whenever the visible window can be reduced quickly.",
    ),
    (
        "Reduce what is read",
        "a written pyramid",
        "Store several resolutions, then read the one that matches the zoom.",
        "Needed once reducing the widest view is too slow, or the data exceeds memory.",
    ),
    (
        "Reduce what is stored",
        "a reference, not a copy",
        "Point at chunks already in the archive instead of duplicating them.",
        "Storage, not display. This tier changes nothing about what is drawn.",
    ),
]


def escalation_diagram(compact=False) -> str:
    """Three tiers, cheapest first, in the order the notebook introduces them.

    Ordered as an escalation rather than as a data path, because the reader's first question
    is whether the cheapest option is already enough. If it is, the later tiers never need to
    be considered at all.
    """
    tier_h = 58 if compact else 84
    w = 700 if compact else 760
    h = 128 + len(TIERS) * (tier_h + 16) + (0 if compact else 34)

    svg = head(w, h, "Three tiers of reduction, cheapest first, each with what it reduces")
    svg += text(24, 30, "Start with the cheapest reduction", size=15, weight="600")
    svg += text(
        24, 48,
        "Each tier is only needed if the one above it is not enough."
        if compact else
        "Each tier is only needed if the one above it is not enough, and this is the order the "
        "sections below follow.",
        size=12, fill=MUTED,
    )

    # The constraint that starts the whole question.
    svg += rect(24, 62, w - 48, 34, fill=ACCENT_SOFT, stroke=ACCENT, width=1.2, radius=4)
    svg += text(
        36, 84,
        "A plot has only as many pixels as its window, perhaps one or two thousand across.",
        size=12, fill=ACCENT, weight="600",
    )

    for index, (title, technique, what, when) in enumerate(TIERS):
        y = 112 + index * (tier_h + 16)
        svg += rect(24, y, w - 48, tier_h, fill="#fbfcfc", stroke=MUTED, width=1.2, radius=4)
        svg += (
            f'<rect x="24" y="{y}" width="5" height="{tier_h}" fill="{ACCENT}" '
            f'fill-opacity="{1 - index * 0.28:.2f}"/>\n'
        )
        svg += text(44, y + 22, f"{index + 1}.", size=12, fill=MUTED, weight="600")
        svg += text(66, y + 22, title, size=13, weight="600")
        svg += text(66 + 178, y + 22, technique, size=12, fill=ACCENT, weight="600")
        if not compact:
            svg += text(66, y + 42, what, size=11, fill=INK)
            svg += text(66, y + 60, when, size=11, fill=MUTED, style="italic")
        else:
            svg += text(66, y + 42, when, size=10, fill=MUTED, style="italic")

        if index < len(TIERS) - 1:
            arrow_y = y + tier_h
            svg += arrow(w / 2, arrow_y + 2, w / 2, arrow_y + 14)
            svg += text(w / 2 + 10, arrow_y + 13, "not enough", size=10, fill=MUTED)

    if not compact:
        svg += text(
            24, h - 14,
            "Only the first tier is a choice between alternatives. Rasterizing and downsampling "
            "do the same job, so pick one.",
            size=12, fill=INK, style="italic",
        )
    return svg + "</svg>\n"


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    for name, builder in (
        ("concept_rasterize", rasterize_diagram),
        ("concept_downsample", downsample_diagram),
        ("concept_pyramid", pyramid_diagram),
        ("concept_escalation", escalation_diagram),
        ("concept_escalation_compact", lambda: escalation_diagram(compact=True)),
    ):
        path = ASSETS / f"{name}.svg"
        path.write_text(builder())
        print(f"{path.name:28} {path.stat().st_size / 1e3:6.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

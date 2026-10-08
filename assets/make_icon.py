"""Scratchboard-style engraving of the mouse head from AppTileSourceImageCandidate1.png -> SVG.

Light lines on a dark ground; line width follows the photo's brightness; lines are displaced by
the smoothed brightness so they bend over the form; a violet rim light on the subject's lit edge.
The frame is a still from the lab's behaviour camera (not in this repo). Variant B is the app icon:

    python make_icon.py FRAME.png OUT_PREFIX     # writes OUT_PREFIX{A,B}.svg
    rsvg-convert -w 512 -h 512 OUT_PREFIXB.svg -o icon.png
"""
import sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

# crop of the original (px): square around the head
CX, CY, CS = 790, 770, 470
TILE0, TILE = 100, 824          # tile position/size in the 1024 canvas
G = 1024                        # canvas

# outline points were read off a 1000 px view of the frame (display px; orig = 560 + 0.8 * d)
def orig(p):
    return (560 + 0.8 * p[0], 520 + 0.8 * p[1])

MOUSE = [(225, 300), (300, 282), (380, 290), (450, 305), (520, 335), (545, 375), (575, 368), (610, 385),
         (632, 425), (645, 480), (660, 515), (710, 518), (760, 530), (800, 555), (825, 600), (838, 660),
         (832, 730), (812, 790), (772, 835), (722, 872), (662, 897), (600, 895), (545, 885), (480, 880),
         (410, 850), (330, 815), (260, 790), (195, 742), (152, 680), (132, 600), (140, 520), (168, 430),
         (198, 352)]
BOARD = [(628, 472), (948, -20), (1020, -20), (1020, 45), (702, 512)]
EYES = [((549, 736), 7.0), ((690, 664), 5.0)]  # catchlight centre (view px), radius (tile px)
CABLES = [((0, 15), (515, 545)), ((128, -10), (548, 522))]


HEAD_SCALE = 0.72


def kymograph():
    """Frog-heart contractions scratched into a smoked drum (after Loewi, 1921): a lever trace whose
    beats shrink and slow during vagal stimulation and then recover, with the signal and time markers
    below. Each upstroke bends back slightly, as the lever's tip writes an arc."""
    base, amp = 846.0, 150.0
    rng = np.random.default_rng(3)
    x, pts = 128.0, []
    stim = (430.0, 515.0)
    while x < 905:
        if x < stim[0]:
            a, period = 1.0, 44.0
        else:
            k = (x - stim[0]) / 330.0            # recovery after the stimulus
            a = 0.28 + 0.72 * min(1.0, max(0.0, k - 0.12) ** 1.4 * 1.3)
            period = 44.0 + 22.0 * (1 - min(1.0, k * 1.4))
        h = amp * a * (1 + 0.04 * rng.standard_normal())
        # upstroke (with lever arc), relaxation, diastole
        for i in range(7):
            t = i / 6
            pts.append((x - 3.0 * t * t * a, base - h * (1 - (1 - t) ** 2)))
        for i in range(1, 15):
            t = i / 14
            pts.append((x + 2 + 20 * t, base - h * (1 - t) ** 2.2))
        pts.append((x + period - 2, base))
        x += period
    pts = [(px + 0.35 * rng.standard_normal(), py + 0.35 * rng.standard_normal()) for px, py in pts]  # hand-made wobble
    d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    sig = (f"M120,872 L{stim[0]:.0f},872 L{stim[0]:.0f},862 L{stim[1]:.0f},862 L{stim[1]:.0f},872 L910,872")
    ticks = "".join(f"M{tx},888 L{tx},881 " for tx in range(150, 900, 30))
    scratch = "#d6cdbb"
    return f"""<g fill="none" stroke-linejoin="round" stroke-linecap="round">
  <path d="{d}" stroke="{scratch}" stroke-width="4.2" opacity="0.9"/>
  <path d="{d}" stroke="#ffffff" stroke-width="1.6" opacity="0.5" transform="translate(0.8 -0.6)"/>
  <path d="{sig}" stroke="{scratch}" stroke-width="3.5" opacity="0.8"/>
  <path d="M120,888 L910,888 {ticks}" stroke="{scratch}" stroke-width="2.5" opacity="0.7"/>
</g>"""


def to_tile(p):
    x, y = orig(p)
    return ((x - CX) / CS * TILE, (y - CY) / CS * TILE)


def main(out):
    im = Image.open(sys.argv[1]).convert("L").crop((CX, CY, CX + CS, CY + CS)).resize((TILE, TILE), Image.LANCZOS)
    L = np.asarray(im, np.float64) / 255.0
    # subject mask (mouse + headstage board + cables), softened
    m = Image.new("L", (TILE, TILE), 0)
    d = ImageDraw.Draw(m)
    d.polygon([to_tile(p) for p in MOUSE], fill=255)
    d.polygon([to_tile(p) for p in BOARD], fill=255)
    for a, b in CABLES:
        d.line([to_tile(a), to_tile(b)], fill=255, width=int(30 * TILE / CS))
    poly = np.asarray(m, np.float64) / 255.0 > 0.5
    # refine the hand-drawn outline with the photo: near the outline, fur (dark) is subject and
    # bedding (bright) is not; well inside, everything is subject (the face blaze is bright too)
    core = ndi.binary_erosion(poly, iterations=22)
    band = ndi.binary_dilation(poly, iterations=14) & ~core
    dark = ndi.gaussian_filter(L, 2.5) < 0.42
    hard = core | (band & dark)
    hard = ndi.binary_opening(hard, iterations=3)
    hard = ndi.binary_fill_holes(ndi.binary_closing(hard, iterations=4))
    lbl, n = ndi.label(hard)
    if n > 1:
        sizes = ndi.sum(hard, lbl, range(1, n + 1))
        hard = np.isin(lbl, 1 + np.flatnonzero(sizes > 2000))
    hard = (ndi.gaussian_filter(hard.astype(np.float64), 5) > 0.5).astype(np.float64)  # smooth outline
    mask = ndi.gaussian_filter(hard, 2.5)

    # tone: subject keeps its tones (gentle S-curve), background sinks to a faint glow
    Ls = ndi.gaussian_filter(L, 1.2)
    tone = np.clip((Ls - 0.10) / 0.85, 0, 1)
    tone = np.clip(tone + 0.7 * (tone - ndi.gaussian_filter(tone, 7)), 0, 1) ** 1.35  # local contrast
    bg = 0.07 * ndi.gaussian_filter(L, 4)
    T = mask * (0.09 + 0.91 * tone) + (1 - mask) * bg  # dark fur keeps hairline strokes

    # rim light: inside the subject's edge, facing the lamp (upper left)
    dist = ndi.distance_transform_edt(hard)
    gy, gx = np.gradient(ndi.gaussian_filter(hard, 6))
    nrm = np.hypot(gx, gy) + 1e-9
    out_x, out_y = -gx / nrm, -gy / nrm            # outward normal
    lamp = np.array([-0.55, -0.83])
    facing = np.clip(out_x * lamp[0] + out_y * lamp[1], 0, 1) ** 1.5
    rim = np.exp(-dist / 7.0) * facing * (hard > 0)
    rim = ndi.gaussian_filter(rim, 1.5)

    # eyes: solid dark pupils (their catchlights are drawn on top)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    for (ex, ey), r in EYES:
        tx, ty = to_tile((ex + 3, ey + 4))
        T = np.where((xx - tx) ** 2 + (yy - ty) ** 2 < (r * 3.0) ** 2, 0.0, T)
    fy = np.clip((1.0 - yy / TILE) / 0.16, 0, 1)
    fx = np.clip((1.0 - xx / TILE) / 0.10, 0, 1)
    T = T * (fy * fx) ** 1.5
    flow = ndi.gaussian_filter(T, 18)               # lines bend over the form

    def sample(a, x, y):
        return ndi.map_coordinates(a, [y, x], order=1, mode="nearest")

    theta = np.deg2rad(-32)
    dvec = np.array([np.cos(theta), np.sin(theta)])
    nvec = np.array([-dvec[1], dvec[0]])
    spacing = 6.0
    wmax = 5.0
    c0 = np.array([TILE / 2, TILE / 2])
    half = TILE * 0.75
    ts = np.arange(-half, half, 1.5)
    grey = np.array([222, 224, 230])
    violet = np.array([160, 120, 255])
    polys = []
    for s in np.arange(-half, half, spacing):
        base = c0[None, :] + ts[:, None] * dvec[None, :] + s * nvec[None, :]
        disp = 12 * (sample(flow, base[:, 0], base[:, 1]) - 0.35)
        p = base + disp[:, None] * nvec[None, :]
        inside = (p[:, 0] > -4) & (p[:, 0] < TILE + 4) & (p[:, 1] > -4) & (p[:, 1] < TILE + 4)
        tv = sample(T, p[:, 0], p[:, 1])
        rv = sample(rim, p[:, 0], p[:, 1])
        w = wmax * tv + 3.6 * rv
        w[~inside] = 0
        # tangent for offsetting
        tan = np.gradient(p, axis=0)
        tan /= np.linalg.norm(tan, axis=1)[:, None] + 1e-9
        nn = np.stack([-tan[:, 1], tan[:, 0]], 1)
        on = w > 0.5
        i = 0
        n = len(ts)
        while i < n:
            if not on[i]:
                i += 1
                continue
            j = i
            while j < n and on[j] and j - i < 7:
                j += 1
            k = min(j + 1, n)  # overlap one sample so chunks join
            seg = slice(i, k)
            left = p[seg] + nn[seg] * (w[seg, None] / 2)
            right = p[seg] - nn[seg] * (w[seg, None] / 2)
            pts = np.concatenate([left, right[::-1]]) + TILE0
            r = float(np.clip(rv[seg].mean() * 2.6, 0, 1))
            col = (grey * (1 - r) + violet * r).astype(int)
            polys.append((pts, "#%02x%02x%02x" % tuple(col)))
            i = j
    body = "\n".join(
        '<path d="M%s Z" fill="%s"/>' % (" L".join("%.1f,%.1f" % (x, y) for x, y in pts), c) for pts, c in polys
    )
    eyes = ""
    for (ex, ey), r in EYES:  # catchlights
        tx, ty = to_tile((ex, ey))
        eyes += f'<circle cx="{tx + TILE0:.1f}" cy="{ty + TILE0:.1f}" r="{r}" fill="#f4f2ff"/>'
    body += eyes
    for variant in ("A", "B"):
        vign = ""
        if variant == "B":
            vign = (
                '<rect x="100" y="100" width="824" height="824" fill="url(#vig)"/>'
            )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{G}" height="{G}" viewBox="0 0 {G} {G}">
<defs>
  <clipPath id="tile"><rect x="100" y="100" width="824" height="824" rx="185"/></clipPath>
  <filter id="soot" x="0" y="0" width="100%" height="100%">
    <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="7"/>
    <feColorMatrix type="matrix" values="0 0 0 0 0.5  0 0 0 0 0.5  0 0 0 0 0.52  0 0 0 1.4 -0.55"/>
  </filter>
  <radialGradient id="ground" cx="0.5" cy="0.55" r="0.7">
    <stop offset="0" stop-color="#22222a"/>
    <stop offset="1" stop-color="#060608"/>
  </radialGradient>
  <radialGradient id="vig" cx="0.5" cy="0.55" r="0.75">
    <stop offset="0.5" stop-color="#000" stop-opacity="0"/>
    <stop offset="1" stop-color="#000" stop-opacity="0.92"/>
  </radialGradient>
</defs>
<g clip-path="url(#tile)">
<rect x="100" y="100" width="824" height="824" fill="{'url(#ground)' if variant == 'B' else '#101014'}"/>
<g transform="translate(100 100) scale({HEAD_SCALE}) translate(-100 -100)">
{body}
</g>
{kymograph()}
{vign}
<rect x="100" y="100" width="824" height="824" filter="url(#soot)" opacity="0.22"/>
</g>
<rect x="100.5" y="100.5" width="823" height="823" rx="184.5" fill="none" stroke="#ffffff" stroke-opacity="0.10" stroke-width="1.5"/>
</svg>
'''
        open(f"{out}{variant}.svg", "w").write(svg)
    print(len(polys), "polygons")


main(sys.argv[2])

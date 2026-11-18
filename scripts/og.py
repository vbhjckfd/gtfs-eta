# /// script
# dependencies = ["pillow"]
# ///
"""Render worker/og.png, the docs page's 1200x630 share image.

    make og              # then: make share-assets, and bump ?v= on og.png in worker/index.html

Fonts are Inter subsets (SIL OFL, see og-fonts/OFL.txt).
"""
import math, pathlib, random
from PIL import Image, ImageDraw, ImageFont
HERE = pathlib.Path(__file__).resolve().parent
F, out = HERE / "og-fonts", HERE.parent / "worker" / "og.png"
W, H = 1200, 630
img = Image.new("RGB", (W, H), "#15171a")
d = ImageDraw.Draw(img)
bold = lambda s: ImageFont.truetype(f"{F}/Inter-800ExtraBold.ttf", s)
med = lambda s: ImageFont.truetype(f"{F}/Inter-500Medium.ttf", s)
# faint route lines on the right
random.seed(4)
cols = ["#0057b7", "#ffd700", "#2f80ed", "#3a3f45"]
for i in range(9):
    y = 80 + i * 58
    pts = [(560 + x, y + 40 * math.sin(x / 140 + i)) for x in range(0, 700, 10)]
    d.line(pts, fill=cols[i % len(cols)] if i % 3 == 0 else "#262a2f", width=6 if i % 3 == 0 else 3)
    if i % 3 == 0:
        for x in (150, 330, 520):
            px, py = 560 + x, y + 40 * math.sin(x / 140 + i)
            d.ellipse((px - 11, py - 11, px + 11, py + 11), fill="#15171a", outline=cols[i % len(cols)], width=5)
# fade the lines under the text
fade = Image.new("L", (W, H), 0)
fd = ImageDraw.Draw(fade)
for x in range(W):
    fd.line([(x, 0), (x, H)], fill=int(255 * max(0, min(1, (820 - x) / 300))))
img.paste(Image.new("RGB", (W, H), "#15171a"), (0, 0), fade)
d = ImageDraw.Draw(img)
# flag bar
d.rectangle((72, 72, 132, 87), fill="#0057b7"); d.rectangle((72, 87, 132, 102), fill="#ffd700")
d.text((150, 69), "eta.lad.lviv.ua", font=med(30), fill="#9aa0a6")
d.text((70, 150), "Lviv GTFS", font=bold(96), fill="#e6e7e8")
d.text((70, 252), "Realtime", font=bold(96), fill="#6aa8ff")
d.text((72, 380), "Open feeds for every tram, trolleybus and bus.", font=med(32), fill="#c9ccd0")
d.text((72, 424), "Arrival times predicted by machine learning.", font=med(32), fill="#c9ccd0")
# stat chips
x = 72
for big, small in (("~93 s", "mean error"), ("10 s", "refresh"), ("free", "no API key")):
    w = max(d.textlength(big, font=bold(40)), d.textlength(small, font=med(22))) + 44
    d.rounded_rectangle((x, 500, x + w, 578), radius=14, fill="#1c1f23", outline="#2c2f33", width=2)
    d.text((x + 22, 507), big, font=bold(40), fill="#e6e7e8")
    d.text((x + 22, 550), small, font=med(22), fill="#9aa0a6")
    x += w + 16
img.save(out, optimize=True)

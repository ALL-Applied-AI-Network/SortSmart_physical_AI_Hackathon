#!/usr/bin/env python3
"""Generate clean, photo-referenced AAA wrapper textures for Isaac Sim.

The source photographs are used only as a visual reference.  Rebuilding the
labels avoids baking the table, shadows and perspective distortion into the
simulated battery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1536
HEIGHT = 512


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _centered(draw: ImageDraw.ImageDraw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    x, y = xy
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text,
              font=font, fill=fill)


def _amazon_pictogram_row(draw: ImageDraw.ImageDraw, y: float) -> None:
    """Draw large regulatory marks that survive a 14 px rendered diameter.

    The real wrapper photographs show bold circular warning symbols around the
    back half of the cell.  Simple vector marks retain that structure after
    Isaac's minification without baking photo lighting into the texture.
    """
    centres = [WIDTH * value for value in (0.34, 0.40, 0.46, 0.52, 0.58, 0.64)]
    radius = 22
    stroke = 6
    for index, x in enumerate(centres):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline="#171717",
            width=stroke,
        )
        if index % 2:
            draw.line(
                (x - 15, y - 15, x + 15, y + 15),
                fill="#171717",
                width=stroke,
            )
        else:
            draw.line(
                (x - 15, y + 15, x + 15, y - 15),
                fill="#171717",
                width=stroke,
            )


def amazon_texture() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#eaa51d")
    draw = ImageDraw.Draw(image)

    # The photographed Amazon cell has blue-black end bands and a warm orange
    # wrapper.  Leave the geometric metal terminals to the USD model.
    band = int(WIDTH * 0.105)
    draw.rectangle((0, 0, band, HEIGHT), fill="#263b48")
    draw.rectangle((WIDTH - band, 0, WIDTH, HEIGHT), fill="#263b48")
    draw.rectangle((band, 0, band + 8, HEIGHT), fill="#d3d7d7")
    draw.rectangle((WIDTH - band - 8, 0, WIDTH - band, HEIGHT), fill="#d3d7d7")

    # Front panel: centred at half-circumference, which faces upward initially.
    _centered(draw, (WIDTH * 0.50, HEIGHT * 0.48), "amazon basics",
              _font(76, bold=False), "#ffffff")
    draw.arc((WIDTH * 0.365, HEIGHT * 0.55, WIDTH * 0.535, HEIGHT * 0.68),
             start=10, end=165, fill="#ffffff", width=9)
    _centered(draw, (WIDTH * 0.50, HEIGHT * 0.18),
              "AAA  LR03  AM4  1.5V", _font(48, bold=True), "#111111")
    _centered(draw, (WIDTH * 0.50, HEIGHT * 0.30),
              "CE     Hg     INSERT CORRECTLY", _font(34, bold=True), "#171717")
    _centered(draw, (WIDTH * 0.50, HEIGHT * 0.71),
              "ALKALINE BATTERY  |  MADE IN CHINA",
              _font(34, bold=True), "#171717")

    # Polarities remain readable even if wrapper roll is randomised.
    _centered(draw, (WIDTH * 0.075, HEIGHT * 0.50), "+", _font(84, True), "#d8edf1")
    _centered(draw, (WIDTH * 0.925, HEIGHT * 0.50), "-", _font(84, True), "#d8edf1")

    # The photographed reverse/side panels contain dense black regulatory
    # print and large warning pictograms.  Cover every circumferential quarter
    # so a random wrapper roll never presents a featureless orange strip.
    warning = "DO NOT RECHARGE   DO NOT DISPOSE OF IN FIRE"
    _centered(draw, (WIDTH * 0.50, HEIGHT * 0.055), warning,
              _font(25, bold=True), "#171717")
    _amazon_pictogram_row(draw, HEIGHT * 0.82)
    _centered(draw, (WIDTH * 0.50, HEIGHT * 0.95),
              "WARNING   OBSERVE POLARITY   DO NOT RECHARGE",
              _font(25, bold=True), "#171717")
    return image


def seven_eleven_texture() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#071622")
    draw = ImageDraw.Draw(image)

    gold = "#c98b23"
    copper = "#c06b32"
    gold_width = int(WIDTH * 0.245)
    draw.rectangle((0, 0, gold_width, HEIGHT), fill=gold)
    draw.rectangle((gold_width - 7, 0, gold_width + 7, HEIGHT), fill="#e4bd63")
    draw.rectangle((WIDTH - 75, 0, WIDTH, HEIGHT), fill="#05090c")

    _centered(draw, (gold_width * 0.48, HEIGHT * 0.48), "24/7 life",
              _font(57, bold=False), "#16110b")
    _centered(draw, (gold_width * 0.52, HEIGHT * 0.76), "BY 7-ELEVEN",
              _font(25, bold=True), "#17110b")
    _centered(draw, (WIDTH * 0.39, HEIGHT * 0.32), "AAA", _font(64, True), copper)
    _centered(draw, (WIDTH * 0.39, HEIGHT * 0.62), "BATTERY  1.5 VOLTS",
              _font(28, True), "#e9ecec")
    _centered(draw, (WIDTH * 0.70, HEIGHT * 0.46), "SUPER", _font(55, True), copper)
    _centered(draw, (WIDTH * 0.70, HEIGHT * 0.67), "ALKALINE PLUS",
              _font(50, True), copper)
    _centered(draw, (WIDTH * 0.65, HEIGHT * 0.10),
              "WARNING: INSERT CORRECTLY  DO NOT RECHARGE", _font(19), "#d4d8d9")
    _centered(draw, (WIDTH * 0.65, HEIGHT * 0.92),
              "0% MERCURY / 0% CADMIUM", _font(21), "#d4d8d9")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="assets/battery_textures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    amazon_texture().save(out / "amazon_basics_aaa.png", optimize=True)
    seven_eleven_texture().save(out / "seven_eleven_aaa.png", optimize=True)
    print(f"Wrote battery textures to {out.resolve()}")


if __name__ == "__main__":
    main()

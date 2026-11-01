"""Host-only checks for the photo-referenced battery texture assets."""

from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parent
TEXTURES = ROOT / "assets" / "battery_textures"


class BatteryBrandTextureTests(unittest.TestCase):
    def test_both_brand_textures_exist_at_expected_size(self):
        for name in ("amazon_basics_aaa.png", "seven_eleven_aaa.png"):
            with self.subTest(name=name):
                with Image.open(TEXTURES / name) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (1536, 512))

    def test_amazon_contains_orange_and_dark_end_bands(self):
        image = Image.open(TEXTURES / "amazon_basics_aaa.png").convert("RGB")
        self.assertLess(sum(image.getpixel((20, 256))), 250)
        centre = image.getpixel((768, 256))
        self.assertGreater(centre[0], 200)
        self.assertGreater(centre[1], 100)

    def test_amazon_has_dark_structure_in_every_circumference_quarter(self):
        image = Image.open(TEXTURES / "amazon_basics_aaa.png").convert("RGB")
        # Ignore end bands; y is the wrapper circumference in the UV layout.
        for y0 in range(0, 512, 128):
            with self.subTest(y0=y0):
                crop = image.crop((170, y0, 1366, y0 + 128))
                pixels = list(crop.get_flattened_data())
                dark_fraction = sum(sum(pixel) < 300 for pixel in pixels) / len(pixels)
                self.assertGreater(dark_fraction, 0.025)

    def test_seven_eleven_contains_gold_and_navy(self):
        image = Image.open(TEXTURES / "seven_eleven_aaa.png").convert("RGB")
        gold = image.getpixel((100, 256))
        navy = image.getpixel((900, 256))
        self.assertGreater(gold[0], 150)
        self.assertLess(sum(navy), 100)


if __name__ == "__main__":
    unittest.main()

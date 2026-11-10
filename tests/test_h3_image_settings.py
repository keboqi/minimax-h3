"""Canvas constraints shared by image generation and prompt assistance."""

import unittest

from h3_app.errors import H3Error
from h3_app.h3_image import H3_IMAGE_EDIT_PIXELS, official_edit_dimensions, validate_h3_image_canvas


class H3ImageCanvasTests(unittest.TestCase):
    def test_matching_a_narrow_source_scales_both_axes_together(self):
        self.assertEqual(official_edit_dimensions(64, 1024, maximize=False), (256, 4096))
        self.assertEqual(official_edit_dimensions(1024, 64, maximize=False), (4096, 256))
        self.assertEqual(official_edit_dimensions(1920, 1080, maximize=False), (1920, 1088))

    def test_maximum_edits_respect_area_and_axis_limits_after_rounding(self):
        for source in ((4, 3), (3, 4), (16, 9), (9, 16), (1, 32), (32, 1), (1, 1)):
            with self.subTest(source=source):
                width, height = official_edit_dimensions(*source, maximize=True)
                validate_h3_image_canvas(width, height)
                self.assertLessEqual(width * height, H3_IMAGE_EDIT_PIXELS)
                self.assertAlmostEqual((width / height) / (source[0] / source[1]), 1, delta=0.02)

    def test_impossible_aspect_ratios_fail_instead_of_changing_shape(self):
        for source in ((1, 100), (100, 1), (0, 1)):
            with self.subTest(source=source), self.assertRaises(H3Error):
                official_edit_dimensions(*source, maximize=False)

    def test_invalid_canvas_values_are_rejected_without_coercion(self):
        for width in (None, "4096", True, float("nan"), float("inf"), 4096.5, 4097, 0):
            with self.subTest(width=width), self.assertRaises(H3Error):
                validate_h3_image_canvas(width, 4096)
        with self.assertRaises(H3Error):
            validate_h3_image_canvas(8192, 8192)


if __name__ == "__main__":
    unittest.main()

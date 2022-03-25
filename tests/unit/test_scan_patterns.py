import unittest

import numpy as np

from logic import scan_patterns


class TestScanPatterns(unittest.TestCase):
    """Doc."""

    def test_calc_ao(self):
        """Doc."""

        set_pnts_lines_odd = np.random.randn(173)
        single_line_ao = np.random.randn(578)
        result = scan_patterns.calculate_image_ao(set_pnts_lines_odd, single_line_ao)
        self.assertEqual(result.shape, (2, set_pnts_lines_odd.size * single_line_ao.size))


if __name__ == "__main__":
    unittest.main()

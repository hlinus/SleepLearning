import unittest
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT_DIR)


# --------------------------------------------------------------------------- #
# ----------------- Define basic functionality tests ------------------------ #
# --------------------------------------------------------------------------- #

class TestBasicFunctionalities(unittest.TestCase):

    def test_sample_data(self):
        pass

    def test_base_sleeplearning_class(self):
        pass


# --------------------------------------------------------------------------- #
# ----------------- Do the testing ------------------------------------------ #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    unittest.main()
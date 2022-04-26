import unittest
from God_hard_to_call import *

class Testlog_barriers(unittest.TestCase):
    def test_0(self):
        x, y = terrifying('1/3*(x1**3+3*x1**3+3*x1+1)+x2', [0, 0], ['-x1+1 <= 0', '-x2 <= 0'], [], [])

        self.assertAlmostEqual(y, 14, places=0)

    def test_1(self):
        x, y = terrifying('x1**2+x2**2', [0, 0], ['-2*x1-x2+2 <= 0'], [], [])

        self.assertAlmostEqual(y, 5, places=0)

    def test_2(self):
        x, y = terrifying('-x1+2*x2**2-4*x2', [0.1, -2], ['3*x1+2*x2+6 <= 0'], [], [])

        self.assertAlmostEqual(y, 0.9, places=0)

    def test_3(self):
        x, y = terrifying('4*x1**2+4*x1+x2**2-8*x2+5', [4.5, -0.75], ['-2*x1+x2-6<=0'], [], [])

        self.assertAlmostEqual(y, -12, places=0)

    def test_4(self):
        x, y = terrifying('8*x1**2-4*x1+x2**2-12*x2+7', [-1.7, -0.5], ['2*x1+3*x2+6<=0'], [], [])

        self.assertAlmostEqual(y, 37, places=0)

    def test_5(self):
        x, y = terrifying('x1 - 2*x2 ', [0.5, 0.5], ['-1-x1+x2^2 <=0', '-x2<=0', 'x1+x2-1=0'], [[1, 1]], [-1])

        self.assertAlmostEqual(y, -2, places=0)

if __name__ == '__main__':
    unittest.main()
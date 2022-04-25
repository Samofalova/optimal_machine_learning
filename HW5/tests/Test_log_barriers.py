import unittest
from log_barriers import *

class Testlog_barriers(unittest.TestCase):
    def test_0(self):
        func = '1/3*(x1+1)**3+x2'
        restrictions = ['x1-1>=0', 'x2>=0']
        y = float(log_barriers(func, restrictions, (0, 0)))

        self.assertAlmostEqual(y, 1/3, places=0)

    def test_1(self):
        func = '(x1+4)**2+(x2-4)**2'
        restrictions = ['-2*x1+x2+2>=0', 'x1>=0', 'x2>=0']
        y = float(log_barriers(func, restrictions, (0, 0)))

        self.assertAlmostEqual(y, 32, places=0)

    def test_2(self):
        func = 'x1**2+2*x2**2-4*x1-5*x2'
        restrictions = ['-x1-x2+1>=0']
        y = float(log_barriers(func, restrictions, (0, 0)))

        self.assertAlmostEqual(y, -4.9, places=0)

    def test_3(self):
        func = '10*x1+2*x2'
        restrictions = ['-x1-x2+13>=0', '-x1+x2+8>=0', 'x1>=0', 'x2>=0']
        y = float(log_barriers(func, restrictions, (0, 0)))

        self.assertAlmostEqual(y, 0, places=0)

    def test_4(self):
        func = '-3*x1-9*x2'
        restrictions = ['-x1-x2+12>=0', 'x1-x2+7>=0', 'x1>=0', 'x2>=0']
        y = float(log_barriers(func, restrictions, (0, 0)))

        self.assertAlmostEqual(y, 0, places=0)

    def test_5(self):
        func = '-5*x1-7*x2-4'
        restrictions = ['-x2+x1+2>=0', '-x1-x2+10>=0', 'x1>=0', 'x2>=0']
        y = float(log_barriers(func, restrictions, (0, 0)))

        self.assertAlmostEqual(y, -4, places=0)

    def test_7(self):
        func = '-3*x1-6*x2-3'
        restrictions = ['-x2+x1+2>=0', '-x1-x2+9>=0', 'x1>=0', 'x2>=0']
        y = float(log_barriers(func, restrictions, (0, 0)))

        self.assertAlmostEqual(y, -3, places=0)

    def test_8(self):
        func = '-11*x1-16*x2-0.1*x1**2-0.12*x2**2+0.22*x1*x2'
        restrictions = ['-3*x1-5*x2+120>=0', '-4*x1-6*x2+150>=0', '-14*x1-12*x2+400>=0', 'x1>=0', 'x2>=0']
        y = float(log_barriers(func, restrictions, (0, 0)))

        self.assertAlmostEqual(y, 0, places=0)

    def test_9(self):
        func = '-(0.08*x1+0.1*x2-3*(0.012*x1**2+0.015*x2**2+2*0.0015*x1*x2))'
        restrictions = ['-x1-x2+1>=0', 'x1>=0', 'x2>=0']
        y = float(log_barriers(func, restrictions, (0, 0)))

        self.assertAlmostEqual(y, 0, places=0)

if __name__ == '__main__':
    unittest.main()
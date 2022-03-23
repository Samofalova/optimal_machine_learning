import unittest
import numpy as np
from sympy import (
    pi, sqrt, tan, sin, cos, log, ln, exp, acos, atan, acot, asin, cot
    )
from ALL import bfgs

class TestBFG(unittest.TestCase):
    def test_0(self):
        def f(x):
            return x**2+5
        def f1(x):
            return 2*x
        res = bfgs(f, f1, x0 = 3)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-0, 0.5)
        self.assertLessEqual(value_f-5, 0.5)

    def test_1(self):
        def f(x):
            return x**3-3*np.sin(x)
        def f1(x):
            return 3*x**2 - 3*np.cos(x)
        res=bfgs(f, f1, x0=0)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-0.82413, 0.5)
        self.assertLessEqual(value_f-(-1.64213), 0.5)

    def test_2(self):
        def f(x):
            return x**4+x**2+x+1
        def f1(x):
            return 4*x**3 + 2*x + 1
        res = bfgs(f, f1, x0=0)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-(-0.38546), 0.5)
        self.assertLessEqual(value_f-0.78520, 0.5)

    def test_3(self):
        def f(x):
            return np.exp(x)+1/x
        def f1(x):
            return np.exp(x) - 1/x**2
        res = bfgs(f, f1, x0=0.5)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-0.70347, 0.5)
        self.assertLessEqual(value_f - 3.4423, 0.5)

    def test_4(self):
        def f(x):
            return x**2-2*x+np.exp(-x)
        def f1(x):
            return 2*x - 2 - np.exp(-x)
        res = bfgs(f, f1, x0=1)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-1.1572, 0.5)
        self.assertLessEqual(value_f-(-0.66092), 0.5)

    def test_5(self):
        def f(x):
            return x*np.sin(x)+2*np.cos(x)
        def f1(x):
            return x*np.cos(x) - np.sin(x)
        res = bfgs(f, f1, x0=-4)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-(-4.49341), 0.5)
        self.assertLessEqual(value_f-(-4.82057), 0.5)

    def test_6(self):
        def f(x):
            return x+1/x**2
        def f1(x):
            return 1 - 2/x**3
        res = bfgs(f, f1, x0=1)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-1.2599, 0.5)
        self.assertLessEqual(value_f-1.8899, 0.5)

    # def test_7(self):
    #     def f(x):
    #         return 10*x*np.log(x)-x**2/2
    #     def f1(x):
    #         return -x + 10*np.log(x) + 10
    #     res = bfgs2(f, f1, x0=1)
    #     point = res['point']
    #     value_f = res['value_func']
    #     self.assertLessEqual(point-0.38221, 0.5)
    #     self.assertLessEqual(value_f+3.7491, 0.5)

    def test_8(self):
        def f(x):
            return np.exp(x)-1/3*x**3+2*x
        def f1(x):
            return -1.0*x**2 + np.exp(x) + 2
        res = bfgs(f, f1, x0=-1)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-(-1.49165), 0.5)
        self.assertLessEqual(value_f-(-1.65198), 0.5)

    def test_9(self):
        def f(x):
            return x**2-2*x-2*np.cos(x)
        def f1(x):
            return 2*x + 2*np.sin(x) - 2
        res = bfgs(f, f1, x0=1)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-0.51097, 0.5)
        self.assertLessEqual(value_f-(-2.50539), 0.5)

    def test_10(self):
        def f(x):
            return np.exp(x)-1-x-x**2/2-x**3/6
        def f1(x):
            return --x**2/2 - x + np.exp(x) - 1
        res = bfgs(f, f1, x0=1)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-0.000, 0.3)
        self.assertLessEqual(value_f-0.000000, 0.5)

    def test_11(self):
        def f(x):
            return x**(np.exp(0.1*x)*np.sin(x))
        def f1(x):
            return x**(np.exp(0.1*x)*np.sin(x))*((0.1*np.exp(0.1*x)*np.sin(x) + np.exp(0.1*x)*np.cos(x))*np.log(x) + np.exp(0.1*x)*np.sin(x)/x)
        res = bfgs(f, f1, x0=8)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-8.01261, 0.5)
        self.assertLessEqual(value_f-97.4179, 0.4)

    def test_12(self):
        def f(x):
            return -5*x**5+4*x**4-12*x**3+11*x**2-2*x+1
        def f1(x):
            return -25*x**4 + 16*x**3 - 36*x**2 + 22*x - 2
        res = bfgs(f,  f1, x0=0.5)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-0.10986, 0.5)
        self.assertLessEqual(value_f-0.897633, 0.5)

    def test_13(self):
        def f(x):
            return -(np.log(x-2))**2 + (np.log(10-x))**2-x**(1/5)
        def f1(x):
            return -0.2*x**(-0.8) - 2*np.log(x - 2)/(x - 2) - 2*np.log(10 - x)/(10 - x)
        res = bfgs(f, f1, x0=9)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-9.20624, 0.5)
        self.assertLessEqual(value_f-(-5.40596), 0.5)

    def test_14(self):
        def f(x):
            return -3*x*np.sin(0.75*x)+np.exp(-2*x)
        def f1(x):
            return -2.25*x*np.cos(0.75*x) - 3*np.sin(0.75*x) - 2*np.exp(-2*x)
        res = bfgs(f, f1, x0=2)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-2.70648, 0.5)
        self.assertLessEqual(value_f-(-7.27436), 0.5)

    def test_15(self):
        def f(x):
            return np.exp(3*x)+5*np.exp(-2*x)
        def f1(x):
            return 3*np.exp(3*x) - 10*np.exp(-2*x)
        res = bfgs(f, f1, x0=1)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-0.24079, 0.5)
        self.assertLessEqual(value_f-5.1483, 0.4)

    def test_16(self):
        def f(x):
            return 0.2*x*np.log(x)+(x-2.3)**2
        def f1(x):
            return 2.0*x + 0.2*np.log(x) - 4.4
        res = bfgs(f, f1, x0=2)
        point = res['point']
        value_f = res['value_func']
        self.assertLessEqual(point-2.1246, 0.4)
        self.assertLessEqual(value_f-0.35098, 0.5)

    # def test_17(self):
    #     def f(x):
    #         return 1/((x-1)**2)*(np.log(x)-2*(x-1)/(x+1))
    #     def f1(x):
    #         return (-2/(x + 1) + (2*x - 2)/(x + 1)**2 + 1/x)/(x - 1)**2 - 2*(np.log(x) - (2*x - 2)/(x + 1))/(x - 1)**3
    #     res = bfgs(f, f1, x0=4)
    #     point = res['point']
    #     value_f = res['value_func']
    #     self.assertLessEqual(point-2.1887, 0.4)
    #     self.assertLessEqual(value_f-(-0.02671), 0.5)

if __name__ == '__main__':
    unittest.main()

import unittest
from sympy import (
    pi, sqrt, tan, sin, cos, log, ln, exp, acos, atan, acot, asin, cot
    )
from ALL import parabola_method

class TestParabolaMethod(unittest.TestCase):
    def test_0(self):
        res = parabola_method(func='X**2+5', limits='[-100, 100]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 0, places=0)
        self.assertAlmostEqual(value_f, 5, places=0)

    def test_1(self):
        res = parabola_method(func='X**3-3*sin(X)', limits='[0, 1]', accuracy=10**(-20))
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 0.8, places=1)
        self.assertAlmostEqual(value_f, -1.64, places=2)

    def test_2(self):
        res = parabola_method(func='X**4+X**2+X+1', limits='[-1, 0]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, -0.4, places=1)
        self.assertAlmostEqual(value_f, 0.79, places=2)

    def test_3(self):
        res = parabola_method(func='exp(X)+1/X', limits='[0.5, 1.5]')
        point = res[0]
        value_f = res[1]

        self.assertAlmostEqual(point, 0.7, places=1)
        self.assertAlmostEqual(value_f, 3.44, places=2)

    def test_4(self):
        res = parabola_method(func='X**2-2*X+exp(-X)', limits='[-1, 1.5]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 1.2, places=1)
        self.assertAlmostEqual(value_f, -0.66, places=2)

    def test_5(self):
        res = parabola_method(func='X*sin(X)+2*cos(X)', limits='[-6, -4]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, -4.5, places=1)
        self.assertAlmostEqual(value_f, -4.82, places=2)

    def test_6(self):
        res = parabola_method(func='X+1/X**2', limits='[1, 2]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 1.3, places=1)
        self.assertAlmostEqual(value_f, 1.89, places=2)

    def test_7(self):
        res = parabola_method(func='10*X*ln(X)-X**2/2', limits='[0.1, 1]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 0.5, places=1)
        self.assertAlmostEqual(value_f, -3.7, places=1)

    def test_8(self):
        res = parabola_method(func='exp(X)-1/3*X**3+2*X', limits='[-2.5, -1]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, -1.5, places=1)
        self.assertAlmostEqual(value_f, -1.65, places=2)

    def test_9(self):
        res = parabola_method(func='X**2-2*X-2*cos(X)', limits='[-0.5, 1]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 0.5, places=1)
        self.assertAlmostEqual(value_f, -2.51, places=2)

    def test_10(self):
        res = parabola_method(func='exp(X)-1-X-X**2/2-X**3/6', limits='[-5, 5]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 0.0, places=1)
        self.assertAlmostEqual(value_f, 0.00, places=2)

    def test_11(self):
        res = parabola_method(func='X**(exp(0.1*X)*sin(X))', limits='[2.5, 14]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 5, places=0)
        self.assertAlmostEqual(value_f, 0.1, places=1)

    def test_12(self):
        res = parabola_method(func='-5*X**5+4*X**4-12*X**3+11*X**2-2*X+1', limits='[-0.5, 0.5]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 0.1, places=1)
        self.assertAlmostEqual(value_f, 0.9, places=1)

    def test_13(self):
        res = parabola_method(func='-(ln(X-2))**2 + (ln(10-X))**2-x**(1/5)', limits='[6, 9.9]')
        point = res[0]
        value_f = res[1]
        if res == 'Выполнено с ошибкой':
            pass
        else:
            self.assertAlmostEqual(point, 9.2, places=1)
            self.assertAlmostEqual(value_f, -5.41, places=2)

    def test_14(self):
        res = parabola_method(func='-3*X*sin(0.75*X)+exp(-2*X)', limits='[0, 2*pi]')
        point = res[0]
        value_f = res[1]
        self.assertGreaterEqual(point, 2)
        self.assertLessEqual(point, 3.5)
        self.assertAlmostEqual(value_f, -7, places=0)

    def test_15(self):
        res = parabola_method(func='exp(3*X)+5*exp(-2*X)', limits='[0, 1]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 0.2, places=1)
        self.assertAlmostEqual(value_f, 5.15, places=2)

    def test_16(self):
        res = parabola_method(func='0.2*X*ln(X)+(X-2.3)**2', limits='[0.5, 2.5]')
        point = res[0]
        value_f = res[1]
        self.assertAlmostEqual(point, 2.1, places=1)
        self.assertAlmostEqual(value_f, 0.35, places=2)

    def test_17(self):
        res = parabola_method(func='-1/((X-1)**2)*(log(X)-2*(X-1)/(X+1))', limits='[1.5, 4.5]', accuracy=10**(-20))
        point = res[0]
        value_f = res[1]
        self.assertGreaterEqual(point, 2.5)
        self.assertLessEqual(point, 4)
        self.assertAlmostEqual(value_f, -0.03, places=2)

if __name__ == '__main__':
    unittest.main()
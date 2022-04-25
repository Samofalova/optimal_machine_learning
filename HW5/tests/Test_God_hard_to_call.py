import unittest
from God_hard_to_call import *

class TestGod_hard_to_call(unittest.TestCase):

    def test_0(self):
        func = '(x1-2)**2 + x2**2'
        eqs = ['x1+x2-4 =0', '-4-2*x1+x2 =0']
        x, y = eq_dual_newton(func, eqs, (0, 0))

        x_ans = np.array([4, 0])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertAlmostEqual(y, 20, places=0)

    def test_1(self):
        func = 'x1 + x2 + 3*x3 + 4*x4'
        eqs = ['5*x1 - 6*x2 + x3 - 2*x4 - 2=0', '11*x1 - 14*x2 + 2*x3 - 5*x4-2=0', 'x1-4=0', 'x4=0', 'x3=0']
        x, y = eq_dual_newton(func, eqs, (0, 0, 0, 0))

        x_ans = np.array([0, 0, 0, 0])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertIn(x[2], x_ans)
        self.assertIn(x[3], x_ans)
        self.assertAlmostEqual(y, 0, places=0)

    def test_2(self):
        func = '3*x1-x2-4*x3+0*x4+0*x5'
        eqs = ['1*x2-x3-x4+1=0', '5*x1-x2-x3+2=0', '8*x1-x2-2*x3+x*5-3=0', 'x1-1=0']
        x, y = eq_dual_newton(func, eqs, (0, 0, 0, 0, 0))

        x_ans = np.array([1, -0.25, -0.25, -8.25, 0])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertIn(x[2], x_ans)
        self.assertIn(x[3], x_ans)
        self.assertIn(x[4], x_ans)
        self.assertAlmostEqual(y, -4.5, places=0)

    def test_3(self):
        func = 'x1**2+x2**2'
        eqs = ['2*x1+x2-2 =0']
        x, y = eq_dual_newton(func, eqs, (0, 0))

        x_ans = np.array([4/5, 2/5])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertAlmostEqual(y, 4/5, places=0)

    def test_4(self):
        func = 'x1*x2*x3'
        eqs = ['x1+x2+x3-1 =0', 'x1**2+x2**2+x3**2-1=0']
        x, y = eq_dual_newton(func, eqs, (0, 0, 0))

        x_ans = np.array([0, 0, 0])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertIn(x[2], x_ans)
        self.assertAlmostEqual(y, 0, places=0)

    def test_5(self):
        func = '-x1+2*x2**2-4*x2'
        eqs = ['-3*x1-2*x2-6=0']
        x, y = eq_dual_newton(func, eqs, (0, 0))

        x_ans = np.array([ 0.83333, -2.55556])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertAlmostEqual(y, 0.61111, places=0)

    def test_7(self):
        func = '4*x1**2+8*x1-x2-3'
        eqs = ['x1+x2+2=0']
        x, y = eq_dual_newton(func, eqs, (0, 0))

        x_ans = np.array([-0.875, -1.125])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertAlmostEqual(y, -6.06250, places=0)

    def test_8(self):
        func = '4*x1**2+4*x1+x2**2-8*x2+5'
        eqs = ['2*x1-x2+6=0']
        x, y = eq_dual_newton(func, eqs, (0, 0))

        x_ans = np.array([ 4.5, -0.75])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertAlmostEqual(y, -11.5, places=0)

    def test_9(self):
        func = '8*x1**2-4*x1+x2**2-12*x2+7'
        eqs = ['-2*x1-3*x2-6=0']
        x, y = eq_dual_newton(func, eqs, (0, 0))

        x_ans = np.array([-1.73685,-0.39473])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertAlmostEqual(y, 33.68421, places=0)

    def test_9(self):
        func = '8*x1**2-4*x1+x2**2-12*x2+7'
        eqs = ['-2*x1-3*x2-6=0']
        x, y = eq_dual_newton(func, eqs, (0, 0))

        x_ans = np.array([-1.73685, -0.39473])
        self.assertIn(x[0], x_ans)
        self.assertIn(x[1], x_ans)
        self.assertAlmostEqual(y, 33.68421, places=0)
if __name__ == '__main__':
    unittest.main()

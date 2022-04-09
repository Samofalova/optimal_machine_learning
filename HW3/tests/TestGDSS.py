import unittest
from gradient import GDSS
import numpy as np

class TestGDSS(unittest.TestCase):

    def test_1(self):
        def f(x, y):
            return x * np.exp(-1 / (x ** 2 + y ** 2)) * np.sin((x ** 2 + y ** 2) ** (-1))

        def pf(x, y):
            return np.array([2 * x ** 2 * np.exp(-1 / (x ** 2 + y ** 2)) * np.sin(1 / (x ** 2 + y ** 2)) / (
                        x ** 2 + y ** 2) ** 2 - 2 * x ** 2 * np.exp(-1 / (x ** 2 + y ** 2)) * np.cos(
                1 / (x ** 2 + y ** 2)) / (x ** 2 + y ** 2) ** 2 + np.exp(-1 / (x ** 2 + y ** 2)) * np.sin(
                1 / (x ** 2 + y ** 2)),
                             2 * x * y * np.exp(-1 / (x ** 2 + y ** 2)) * np.sin(1 / (x ** 2 + y ** 2)) / (
                                         x ** 2 + y ** 2) ** 2 - 2 * x * y * np.exp(-1 / (x ** 2 + y ** 2)) * np.cos(
                                 1 / (x ** 2 + y ** 2)) / (x ** 2 + y ** 2) ** 2])

        res = GDSS(f, pf, lr_rate=0.0005, dataset_rec=1, accuracy=10 ** (-10))
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, 0, places=0)

    def test_2(self):
        def f(x, y):
            return -20*np.exp(-0.2*(0.5*x**2+y**2)**0.5)-np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.exp(1) + 20

        def pf(x, y):
            return np.array([2.0 * x * (0.5 * x ** 2 + y ** 2) ** (-0.5) * np.exp(
                -0.2 * (0.5 * x ** 2 + y ** 2) ** 0.5) + 3.14159265358979 * np.exp(
                0.5 * np.cos(6.28318530717959 * x) + 0.5 * np.cos(6.28318530717959 * y)) * np.sin(6.28318530717959 * x),
                             4.0 * y * (0.5 * x ** 2 + y ** 2) ** (-0.5) * np.exp(
                                 -0.2 * (0.5 * x ** 2 + y ** 2) ** 0.5) + 3.14159265358979 * np.exp(
                                 0.5 * np.cos(6.28318530717959 * x) + 0.5 * np.cos(6.28318530717959 * y)) * np.sin(
                                 6.28318530717959 * y)])

        res = GDSS(f, pf, lr_rate=0.0005, dataset_rec=1, accuracy=10 ** -10)
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, 0, places=0)

    def test_3(self):
        def f(x, y):
            return x ** 2 + y ** 2

        def pf(x, y):
            return np.array([2 * x,
                             2 * y])

        res = GDSS(f, pf, lr_rate=0.05, dataset_rec=1, accuracy=10 ** (-10))
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, 0, places=0)

    def test_4(self):
        def f(x, y):
            return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

        def pf(x, y):
            return np.array([0.52 * x - 0.48 * y,
                             -0.48 * x + 0.52 * y])

        res = GDSS(f, pf, lr_rate=0.05, dataset_rec=1, accuracy=10 ** (-10))
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, 0, places=0)

    def test_5(self):
        def f(x, y):
            return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))

        def pf(x, y):
            return np.array([-(-2 * x + 2 * np.pi) * np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2) * np.cos(x) * np.cos(
                y) + np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2) * np.sin(x) * np.cos(y),
                             -(-2 * y + 2 * np.pi) * np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2) * np.cos(x) * np.cos(
                                 y) + np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2) * np.sin(y) * np.cos(x)])

        res = GDSS(f, pf, lr_rate=0.05, dataset_rec=1, accuracy=10 ** (-10))
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, 0, places=0)

    def test_6(self):
        def f(x, y):
            return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

        def pf(x, y):
            return np.array([4 * x * (x ** 2 + y - 11) + 2 * x + 2 * y ** 2 - 14,
                             2 * x ** 2 + 4 * y * (x + y ** 2 - 7) + 2 * y - 22])

        res = GDSS(f, pf, lr_rate=0.01, dataset_rec=1, accuracy=10 ** (-5))
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, 0, places=0)

    def test_7(self):
        def f(x, y):
            return (x + 2*y -7)**2 + (2*x + y - 5)**2

        def pf(x, y):
            return np.array([10 * x + 8 * y - 34,
                             8 * x + 10 * y - 38])

        res = GDSS(f, pf, lr_rate=0.01, dataset_rec=1, accuracy=10 ** (-10))
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, 0, places=0)

    def test_8(self):
        def f(x, y):
            return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 2) ** 2

        def pf(x, y):
            return np.array([2.25 * (1.33333333333333 * y - 1.33333333333333) * (
                        0.666666666666667 * x * y - 0.666666666666667 * x + 1) + 6.890625 * (
                                         0.761904761904762 * y ** 2 - 0.761904761904762) * (
                                         0.380952380952381 * x * y ** 2 - 0.380952380952381 * x + 1) + 5.0625 * (
                                         0.888888888888889 * y ** 2 - 0.888888888888889) * (
                                         0.444444444444444 * x * y ** 2 - 0.444444444444444 * x + 1),
                             10.5 * x * y * (
                                         0.380952380952381 * x * y ** 2 - 0.380952380952381 * x + 1) + 9.0 * x * y * (
                                         0.444444444444444 * x * y ** 2 - 0.444444444444444 * x + 1) + 3.0 * x * (
                                         0.666666666666667 * x * y - 0.666666666666667 * x + 1)])

        res = GDSS(f, pf, lr_rate=0.01, dataset_rec=1, accuracy=10 ** (-10))
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, 0, places=0)

    def test_9(self):
        def f(x, y, z):
            return 10 * 3 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y)) + (
                        z ** 2 - 10 * np.cos(2 * np.pi * z))

        def pf(x, y, z):
            return np.array([2 * x + 20 * np.pi * np.sin(2 * np.pi * x),
                             2 * y + 20 * np.pi * np.sin(2 * np.pi * y),
                             2 * z + 20 * np.pi * np.sin(2 * np.pi * z)])

        res = GDSS(f, pf, lr_rate=0.5, dataset_rec=1, accuracy=10**(-10))
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, 0, places=0)

    def test_10(self):
        def f(x, y):
            return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

        def pf(x, y):
            return np.array([2 * x - 2 * y + np.cos(x + y) - 1.5,
                             -2 * x + 2 * y + np.cos(x + y) + 2.5])

        res = GDSS(f, pf, lr_rate=0.1, dataset_rec=1, accuracy=10 ** (-10))
        value_f = res['interim_results_dataset']['f'].iloc[-1]
        self.assertAlmostEqual(value_f, -1.9, places=1)

if __name__ == '__main__':
    unittest.main()

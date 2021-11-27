
import sys
sys.path.append('..')

from evaluate.metrics import main as metrics
import io
import unittest



class TestMetricsMethods(unittest.TestCase):
    """
    def test_metrics_one_user(self):

        args = [
            '-g', 'gt.json',
            '-p', 'tracking_predict_1.json',
            '-o', 'output']
        metrics(args)

        with io.open('output/metrics.csv') as result:
            with io.open('expect/expect_metrics_1.csv') as expect:
                self.assertListEqual(list(result), list(expect))

    """
    def test_metrics_one_user_error_predict(self):

        args = [
            '-g', 'gt.json',
            '-p', 'tracking_predict_2.json',
            '-o', 'output']
        metrics(args)

        with io.open('output/metrics.csv') as result:
            with io.open('expect/expect_metrics_2.csv') as expect:
                self.assertListEqual(list(result), list(expect))


if __name__ == '__main__':
    unittest.main()

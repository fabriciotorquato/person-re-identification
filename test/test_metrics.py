
import sys
sys.path.append('../src')

from evaluate.metrics import get_recall, get_precission, get_f_score,get_accuracy_recognition
from evaluate.metrics import main as metrics
import io
import unittest



class TestMetricsMethods(unittest.TestCase):

    def test_metrics_one_user(self):

        args = [
            '-g', 'gt.json',
            '-p', 'tracking_predict_1.json',
            '-o', 'output']
        metrics(args)

        with io.open('output/metrics.csv') as result:
            with io.open('expect/expect_metrics_1.csv') as expect:
                self.assertListEqual(list(result), list(expect))

    def test_metrics_one_user_error_predict(self):

        args = [
            '-g', 'gt.json',
            '-p', 'tracking_predict_2.json',
            '-o', 'output']
        metrics(args)

        with io.open('output/metrics.csv') as result:
            with io.open('expect/expect_metrics_2.csv') as expect:
                self.assertListEqual(list(result), list(expect))

    def test_metrics_many_user_error_predict(self):

        args = [
            '-g', 'gt.json',
            '-p', 'tracking_predict_3.json',
            '-o', 'output']
        metrics(args)

        with io.open('output/metrics.csv') as result:
            with io.open('expect/expect_metrics_3.csv') as expect:
                self.assertListEqual(list(result), list(expect))

    def test_classification_report(self):
        current_recall = get_recall(90, 10)
        expect_recall = .9
        self.assertEqual(current_recall, expect_recall)
        current_precission = get_precission(90, 30)
        expect_precission = .75
        self.assertEqual(current_precission, expect_precission)
        current_f_score = get_f_score(0.633, 0.95)
        expect_f_score = 0.76
        self.assertEqual(current_f_score, expect_f_score)
        current_accuracy = get_accuracy_recognition(20, 20, 10, 10)
        expect_accuracy = 0.33
        self.assertEqual(current_accuracy, expect_accuracy)

if __name__ == '__main__':
    unittest.main()

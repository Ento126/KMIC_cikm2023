import metric
import torch
import unittest


class TestHitAtK(unittest.TestCase):
    def test_one_element_batch_with_last_element_correct(self):
        predictions = torch.tensor([[0.1, 0.0, 0.4, 0.9]])
        ground_truth_idx = torch.tensor([0])
        expected = 1

        actual = metric.hit_at_k(predictions, ground_truth_idx, k=2, device=torch.device('cpu'))
        self.assertEqual(expected, actual)

    def test_multiple_elements_batch(self):
        predictions = torch.tensor([[0.1, 0.0, 0.4, 0.9],
                                    [0.0, 0.9, 0.1, 1.0],
                                    [0.0, 0.1, 0.2, 0.8]])
        k = 2
        ground_truth_idx = torch.tensor([[1], [2], [3]])
        expected = 2
        actual = metric.hit_at_k(predictions, ground_truth_idx, k=k, device=torch.device('cpu'))
        self.assertEqual(expected, actual)


class TestMrr(unittest.TestCase):
    def test_top_element_is_predicted(self):
        predictions = torch.tensor([[0.3, 0.1, 0.0, 0.2]])
        ground_truth_idx = torch.tensor([2])
        expected_score = 1.0
        actual_score = metric.mrr(predictions, ground_truth_idx)
        self.assertEqual(expected_score, actual_score)

    def test_missing_ground_truth_idx(self):
        predictions = torch.tensor([[0.0, 0.1, 0.2, 0.3],
                                    [0.0, 0.1, 0.2, 0.3]])
        ground_truth_idx = torch.tensor([[4], [4]])
        expected_score = 0.0
        actual_score = metric.mrr(predictions, ground_truth_idx)
        self.assertEqual(expected_score, actual_score)

    def test_multiple_element_batch(self):
        predictions = torch.tensor([[0.0, 0.1, 0.3, 0.2],
                                    [0.0, 0.1, 0.2, 0.3],
                                    [0.3, 0.1, 0.2, 0.0]])
        ground_truth_idx = torch.tensor([[2], [2], [1]])
        expected_score = 0.25 + 1/3 + 0.5
        actual_score = metric.mrr(predictions, ground_truth_idx)
        self.assertAlmostEqual(expected_score, actual_score)


if __name__ == '__main__':
    unittest.main()

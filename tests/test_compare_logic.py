import unittest


def pick_winner(q_perplexity, b_perplexity):
    return "qmorph" if q_perplexity <= b_perplexity else "baseline"


class CompareLogicTest(unittest.TestCase):
    def test_qmorph_wins_on_lower_perplexity(self):
        self.assertEqual(pick_winner(45.0, 60.0), "qmorph")

    def test_baseline_wins_on_lower_perplexity(self):
        self.assertEqual(pick_winner(75.0, 70.0), "baseline")


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from src.model.classic_embedding import ClassicEmbedding


class ClassicEmbeddingTest(unittest.TestCase):
    def test_output_shape(self):
        model = ClassicEmbedding(vocab_size=50, embed_dim=16, pad_idx=0)
        inputs = torch.tensor([[1, 2, 3], [4, 0, 5]], dtype=torch.long)
        vectors, probs = model(inputs)
        self.assertIsNone(probs)
        self.assertEqual(tuple(vectors.shape), (2, 3, 16))


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from gemini_torch.utils import ImageToTextEmbeddings


class TestImageToTextEmbeddings(unittest.TestCase):
    def setUp(self):
        self.model = ImageToTextEmbeddings(
            patches=16,
            patch_size=16,
            transformer_dim=512,
            img_channels=3,
            seq_len=128,
            reduced_dim=128,
        )

    def test_initialization(self):
        self.assertIsInstance(self.model, ImageToTextEmbeddings)

    def test_forward_with_img_256(self):
        img = torch.randn(1, 3, 256, 256)
        output = self.model(img)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 128, 128))

    def test_forward_with_img_512(self):
        img = torch.randn(1, 3, 512, 512)
        output = self.model(img)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 128, 128))

    def test_forward_with_img_1024(self):
        img = torch.randn(1, 3, 1024, 1024)
        output = self.model(img)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 128, 128))

    def test_forward_with_img_2048(self):
        img = torch.randn(1, 3, 2048, 2048)
        output = self.model(img)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 128, 128))


if __name__ == "__main__":
    unittest.main()

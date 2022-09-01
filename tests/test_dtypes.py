
from unittest import TestCase, main

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..keras_generators.dtypes import Num1DVecSeq, ScalarVecSeq


class TestScaledNum1DVecSeq(TestCase):
    def test_get_origianl_tensors(self):
        data = np.arange(24).reshape((2, 3, 4))
        seq = Num1DVecSeq(name="test", tensors=data)
        scaler = StandardScaler()
        scaled = seq.scale(scaler)
        np.testing.assert_allclose(scaled.norm_na[:, :, 0].mean(), 0.0, atol=1e-7)
        np.testing.assert_allclose(scaled.norm_na[:, :, 0].std(), 1.0, atol=1e-7)
        unscaled = scaled.get_original_tensors()
        np.testing.assert_allclose(unscaled, data)


class TestScaledNumScalarVecSeq(TestCase):
    def test_get_original_tensors(self):
        data = np.arange(6).reshape((2, 3))
        seq = ScalarVecSeq(name="test", tensors=data)
        scaler = StandardScaler()
        scaled = seq.scale(scaler)
        self.assertEqual(scaled.norm_na[:, 0].mean(), 0.0)
        self.assertEqual(scaled.norm_na[:, 0].std(), 1.0)
        uscaled = scaled.get_original_tensors()
        np.testing.assert_allclose(uscaled, data)


if __name__ == '__main__':
    main()



#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

from unittest import TestCase

import numpy as np
from sklearn.preprocessing import StandardScaler

from keras_generators.encoders import ScaleEncoder, PeriodEncoder


class TestScaleEncoder(TestCase):
    def test_fit_encode_decode_tabular(self):
        na = np.arange(10).reshape((5, 2))
        enc = ScaleEncoder(scaler=StandardScaler())
        ena = enc.fit_encode(na)
        expected_ena = np.array(
            [
                [-1.41421356, -1.41421356],
                [-0.70710678, -0.70710678],
                [0.0, 0.0],
                [0.70710678, 0.70710678],
                [1.41421356, 1.41421356],
            ]
        )
        np.array_equal(ena, expected_ena)
        decoded_na = enc.decode(ena)
        self.assertTrue(np.array_equal(decoded_na, na))

    def test_fit_encode_decode__1d_vectors(self):
        na = np.arange(30).reshape((5, 3, 2))
        enc = ScaleEncoder(scaler=StandardScaler())
        ena = enc.fit_encode(na)
        expected_ena = np.array(
            [
                [
                    [-1.62018517, -1.62018517],
                    [-1.38873015, -1.38873015],
                    [-1.15727512, -1.15727512],
                ],
                [
                    [-0.9258201, -0.9258201],
                    [-0.69436507, -0.69436507],
                    [-0.46291005, -0.46291005],
                ],
                [[-0.23145502, -0.23145502], [0.0, 0.0], [0.23145502, 0.23145502]],
                [
                    [0.46291005, 0.46291005],
                    [0.69436507, 0.69436507],
                    [0.9258201, 0.9258201],
                ],
                [
                    [1.15727512, 1.15727512],
                    [1.38873015, 1.38873015],
                    [1.62018517, 1.62018517],
                ],
            ]
        )
        np.array_equal(ena, expected_ena)
        decoded_na = enc.decode(ena)
        self.assertTrue(np.array_equal(decoded_na, na))

    def test_fit_encode_decode_continuous_timeseries(self):
        """This case is identical with the TabularData case"""
        pass

    def test_encode_decode_tabular(self):
        na = np.arange(20).reshape((10, 2))
        train, test = na[:5], na[5:]
        enc = ScaleEncoder(scaler=StandardScaler())
        ena = enc.fit_encode(train)
        decoded_na = enc.decode(ena)
        np.array_equal(decoded_na, train)
        ena = enc.encode(test)
        expected_ena = np.array(
            [
                [2.12132034, 2.12132034],
                [2.82842712, 2.82842712],
                [3.53553391, 3.53553391],
                [4.24264069, 4.24264069],
                [4.94974747, 4.94974747],
            ]
        )
        np.array_equal(ena, expected_ena)
        decoded_na = enc.decode(ena)
        self.assertTrue(np.array_equal(decoded_na, test))


class TestPeriodEncoder(TestCase):
    def test_fit_encode_tabular(self):
        na = np.arange(10).reshape((5, 2))
        enc = PeriodEncoder(period=8)
        ena = enc.fit_encode(na)
        sin45 = np.sqrt(2) / 2
        expected_ena = np.array(
            [
                [0.0, sin45, 1.0, sin45],
                [1.0, sin45, 0, -sin45],
                [0, -sin45, -1.0, -sin45],
                [-1.0, -sin45, 0, sin45],
                [0.0, sin45, 1.0, sin45],
            ]
        )
        np.testing.assert_allclose(ena, expected_ena, atol=1e-15)

    def test_fit_encode_decode__1d_vectors(self):
        na = np.arange(30).reshape((5, 3, 2))
        enc = PeriodEncoder(period=8)
        ena = enc.fit_encode(na)
        sin45 = np.sqrt(2) / 2
        expected_ena = np.array(
            [
                [
                    [0.0, sin45, 1.0, sin45],
                    [1.0, sin45, 0.0, -sin45],
                    [0, -sin45, -1.0, -sin45],
                ],
                [
                    [-1.0, -sin45, -0.0, sin45],
                    [0.0, sin45, 1.0, sin45],
                    [1.0, sin45, 0.0, -sin45],
                ],
                [
                    [0, -sin45, -1.0, -sin45],
                    [-1.0, -sin45, -0.0, sin45],
                    [0.0, sin45, 1.0, sin45],
                ],
                [
                    [1.0, sin45, 0.0, -sin45],
                    [0, -sin45, -1.0, -sin45],
                    [-1.0, -sin45, -0.0, sin45],
                ],
                [
                    [0.0, sin45, 1.0, sin45],
                    [1.0, sin45, 0.0, -sin45],
                    [0, -sin45, -1.0, -sin45],
                ],
            ]
        )
        np.testing.assert_allclose(ena, expected_ena, atol=1e-15)

    def test_fit_encode_decode_continuous_timeseries(self):
        pass

    def test_encode_decode_tabular(self):
        pass

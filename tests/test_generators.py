#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

from unittest import TestCase, main

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras_generators.encoders import ScaleEncoder
from keras_generators.generators import TensorDataSource, DataSet, TimeseriesDataSource, TimeseriesTargetsParams
from keras_generators.splitters import OrderedSplitter


class TestDataSet(TestCase):
    def test_split_encode(self):
        ds1 = np.arange(20).reshape((10, 2))
        dsource1 = TensorDataSource('ds1', ds1)
        dsource2 = TensorDataSource('ds2', np.arange(70).reshape((10, 7)))
        dsource3 = TensorDataSource('ds3', np.arange(60).reshape((10, 3, 2)))
        dsources = {ds.name: ds for ds in [dsource1, dsource2, dsource3]}
        encoders = {'ds1': [ScaleEncoder(scaler=StandardScaler())],
                    'ds2': [ScaleEncoder(scaler=StandardScaler())],
                    'ds3': [ScaleEncoder(scaler=StandardScaler())]
                    }
        ds_test = DataSet(input_sources=dsources)
        splitter = OrderedSplitter()
        train, val, test = ds_test.split_encode(splitter, encoders=encoders)
        ds1_orig = np.array(train.input_sources['ds1'].decode())

        self.assertTrue(np.array_equal(ds1_orig, ds1[:6]))


class TestTimeseriesDataSource(TestCase):
    def test_split(self):
        splitter = OrderedSplitter()
        rows, cols = 14, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TimeseriesDataSource("test1", na, length=5, stride=1, sampling_rate=1)
        self.assertEqual(len(tds), 10)
        train_ds, val_ds, test_ds = tds.split(splitter)
        self.assertEqual(len(train_ds), 6)
        self.assertEqual(len(val_ds), 2)
        self.assertEqual(len(test_ds), 2)
        expected_train = np.array(
            [[0, 1],
             [2, 3],
             [4, 5],
             [6, 7],
             [8, 9]])
        self.assertTrue(np.array_equal(train_ds[0], expected_train))
        expected_last_train = np.array(
            [[10, 11],
             [12, 13],
             [14, 15],
             [16, 17],
             [18, 19]])
        self.assertTrue(np.array_equal(train_ds[-1], expected_last_train))
        expected_val = np.array(
            [[12, 13],
             [14, 15],
             [16, 17],
             [18, 19],
             [20, 21]])
        self.assertTrue(np.array_equal(val_ds[0], expected_val))
        expected_test = np.array(
            [[16, 17],
             [18, 19],
             [20, 21],
             [22, 23],
             [24, 25]])
        self.assertTrue(np.array_equal(test_ds[0], expected_test))

    def test_split_with_stride(self):
        splitter = OrderedSplitter()
        rows, cols = 14, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TimeseriesDataSource("test1", na, length=5, stride=2, sampling_rate=1)
        self.assertEqual(len(tds), 5)
        train_ds, val_ds, test_ds = tds.split(splitter)
        self.assertEqual(len(train_ds), 3)
        self.assertEqual(len(val_ds), 1)
        self.assertEqual(len(test_ds), 1)
        self.assertTrue(np.array_equal(train_ds[0], na[:5]))
        self.assertTrue(np.array_equal(train_ds[1], na[2:7]))
        self.assertTrue(np.array_equal(train_ds[2], na[4:9]))
        self.assertTrue(np.array_equal(val_ds[0], na[6:11]))
        self.assertTrue(np.array_equal(test_ds[0], na[8:13]))

    def test_split_with_sampling_rate(self):
        splitter = OrderedSplitter()
        rows, cols = 14, 2
        sr = 2
        l = 5
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TimeseriesDataSource("test1", na, length=l, stride=1, sampling_rate=sr)
        self.assertEqual(len(tds), 6)
        train_ds, val_ds, test_ds = tds.split(splitter)
        self.assertEqual(len(train_ds), 3)
        self.assertEqual(len(val_ds), 1)
        self.assertEqual(len(test_ds), 2)
        self.assertTrue(np.array_equal(train_ds[0], na[:sr * l:sr]))

    def test_convert_negative_index(self):
        rows, cols = 15, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TimeseriesDataSource("test1", na, length=5, stride=3, sampling_rate=2)
        self.assertEqual(len(tds), 3)
        self.assertTrue(np.array_equal(tds[-1], tds[2]))

    def test_len(self):
        # Test boundary between 2 and 3
        rows, cols = 14, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TimeseriesDataSource("test1", na, length=5, stride=3, sampling_rate=2)
        self.assertEqual(len(tds), 2)
        rows, cols = 15, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TimeseriesDataSource("test1", na, length=5, stride=3, sampling_rate=2)
        self.assertEqual(len(tds), 3)

        # test boundaries between 1 and 2 with bigger sampling rate == 3
        rows, cols = 13, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TimeseriesDataSource("test1", na, length=5, stride=4, sampling_rate=3)
        self.assertEqual(len(tds), 1)
        rows, cols = 16, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TimeseriesDataSource("test1", na, length=5, stride=4, sampling_rate=3)
        self.assertEqual(len(tds), 1)
        rows, cols = 17, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TimeseriesDataSource("test1", na, length=5, stride=4, sampling_rate=3)
        self.assertEqual(len(tds), 2)

    def test_get_one_instance(self):
        # LSTM uses  [batch, timesteps, feature].
        # we'd expect (batch, timesteps, feature)'
        na = np.arange(20).reshape((10, 2))
        tds = TimeseriesDataSource("test1", na, length=5)
        self.assertEqual(len(tds), 6)
        it1 = tds[0]
        expected_it1 = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        self.assertTrue(np.array_equal(it1, expected_it1))
        expected_last_it = np.array([[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]])
        last_it = tds[-1]
        self.assertTrue(np.array_equal(last_it, expected_last_it))

    def test_get_slice(self):
        na = np.arange(20).reshape((10, 2))
        tds = TimeseriesDataSource("test1", na, length=5)
        expected_slice = np.array(
            [[[2., 3.],
              [4., 5.],
              [6., 7.],
              [8., 9.],
              [10., 11.]],
             [[4., 5.],
              [6., 7.],
              [8., 9.],
              [10., 11.],
              [12., 13.]]])
        slice = tds[1:3]
        self.assertTrue(np.array_equal(slice, expected_slice))

    def test_encode(self):
        na = np.arange(20).reshape((10, 2))
        na[:, 1] = 0
        tds = TimeseriesDataSource("test1", na, length=5)
        encoded = tds.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])

        expected_encoded = np.array(
            [[0., 0.],
             [0.11111111, 0.],
             [0.22222222, 0.],
             [0.33333333, 0.],
             [0.44444444, 0.]], )
        true_encoded = encoded[0]
        np.testing.assert_array_almost_equal(true_encoded, expected_encoded)
        decoded = encoded.decode()
        np.testing.assert_array_almost_equal(decoded[0], na[:5])

    def test_get_targets(self):
        rows, cols = 15, 3
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 1] = 0
        na[:, 0] = np.arange(rows)
        target_params = TimeseriesTargetsParams(delay=1, pred_len=3, stride=2)
        tds = TimeseriesDataSource("test1", na, length=5, stride=2, target_params=target_params)
        targets = tds.get_targets()
        self.assertEqual(targets.shape, (3, 3, 3))
        expected_targets = np.array(
            [[[6, 0, 20],
              [8, 0, 26],
              [10, 0, 32]],
             [[8, 0, 26],
              [10, 0, 32],
              [12, 0, 38]],
             [[10, 0, 32],
              [12, 0, 38],
              [14, 0, 44]]])
        self.assertTrue(np.array_equal(targets, expected_targets))

    def test_split_with_targets_as_orig(self):
        splitter = OrderedSplitter(train=0.6, val=0.2)
        rows, cols = 20, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 0] = np.arange(rows)
        target_params = TimeseriesTargetsParams(delay=1, pred_len=3, stride=2)
        tds = TimeseriesDataSource("test1", na, length=5, stride=1, sampling_rate=1, target_params=target_params)
        self.assertEqual(len(tds), 10)
        train_ds, val_ds, test_ds = tds.split(splitter)
        self.assertEqual(len(train_ds), 6)
        self.assertEqual(len(val_ds), 2)
        self.assertEqual(len(test_ds), 2)
        expected_train = np.array(
            [[0, 1],
             [1, 3],
             [2, 5],
             [3, 7],
             [4, 9]])
        self.assertTrue(np.array_equal(train_ds[0], expected_train))
        expected_last_train = np.array(
            [[5, 11],
             [6, 13],
             [7, 15],
             [8, 17],
             [9, 19]])
        self.assertTrue(np.array_equal(train_ds[-1], expected_last_train))
        expected_val = np.array(
            [[6, 13],
             [7, 15],
             [8, 17],
             [9, 19],
             [10, 21]])
        self.assertTrue(np.array_equal(val_ds[0], expected_val))
        expected_test = np.array(
            [[8, 17],
             [9, 19],
             [10, 21],
             [11, 23],
             [12, 25]])
        self.assertTrue(np.array_equal(test_ds[0], expected_test))


class TestTensorDataSource(TestCase):
    def test_fit_encode(self):
        na = np.array(
            [[[2., 3.],
              [4., 5.],
              [6., 7.],
              [8., 9.],
              [10., 11.]],
             [[4., 5.],
              [6., 7.],
              [8., 9.],
              [10., 11.],
              [12., 13.]]])
        tds = TensorDataSource("test1", na)
        encoded = tds.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        true_encoded = encoded[0]
        expected_encoded = np.array(
            [[0., 0.],
             [0.2, 0.2],
             [0.4, 0.4],
             [0.6, 0.6],
             [0.8, 0.8]])
        np.testing.assert_array_almost_equal(true_encoded, expected_encoded)
        decoded = encoded.decode()
        np.testing.assert_array_almost_equal(decoded[0], na[0])


if __name__ == '__main__':
    main()

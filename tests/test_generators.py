#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

from unittest import TestCase, main

import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras_generators.encoders import ScaleEncoder
from keras_generators.generators import (
    CompoundDataSource,
    DataSet,
    TargetTimeseriesDataSource,
    TensorDataSource,
    TimeseriesDataSource,
    TimeseriesTargetsParams,
    XYBatchGenerator,
    XYWBatchGenerator,
)
from keras_generators.splitters import OrderedSplitter


class TestDataSet(TestCase):
    def test_split_encode(self):
        ds1 = np.arange(20).reshape((10, 2))
        dsource1 = TensorDataSource("ds1", ds1)
        dsource2 = TensorDataSource("ds2", np.arange(70).reshape((10, 7)))
        dsource3 = TensorDataSource("ds3", np.arange(60).reshape((10, 3, 2)))
        dsources = {ds.name: ds for ds in [dsource1, dsource2, dsource3]}
        encoders = {
            "ds1": [ScaleEncoder(scaler=StandardScaler())],
            "ds2": [ScaleEncoder(scaler=StandardScaler())],
            "ds3": [ScaleEncoder(scaler=StandardScaler())],
        }
        ds_test = DataSet(input_sources=dsources)
        splitter = OrderedSplitter()
        train, val, test = ds_test.split_encode(splitter, encoders=encoders)
        ds1_train_orig = np.array(train.input_sources["ds1"].decode())
        np.testing.assert_array_almost_equal(ds1_train_orig, ds1[:6])
        ds1_val_orig = np.array(val.input_sources["ds1"].decode())
        np.testing.assert_array_almost_equal(ds1_val_orig, ds1[6:8])
        ds1_test_orig = np.array(test.input_sources["ds1"].decode())
        np.testing.assert_array_almost_equal(ds1_test_orig, ds1[8:])
        ds2_train_orig = np.array(train.input_sources["ds2"].decode())
        np.testing.assert_array_almost_equal(ds2_train_orig, dsource2.tensors[:6])
        ds2_val_orig = np.array(val.input_sources["ds2"].decode())
        np.testing.assert_array_almost_equal(ds2_val_orig, dsource2.tensors[6:8])
        ds2_test_orig = np.array(test.input_sources["ds2"].decode())
        np.testing.assert_array_almost_equal(ds2_test_orig, dsource2.tensors[8:])
        ds3_train_orig = np.array(train.input_sources["ds3"].decode())
        np.testing.assert_array_almost_equal(ds3_train_orig, dsource3.tensors[:6])
        ds3_val_orig = np.array(val.input_sources["ds3"].decode())
        np.testing.assert_array_almost_equal(ds3_val_orig, dsource3.tensors[6:8])
        ds3_test_orig = np.array(test.input_sources["ds3"].decode())
        np.testing.assert_array_almost_equal(ds3_test_orig, dsource3.tensors[8:])

    def test_split_encode_reversed(self):
        ds1 = np.arange(20).reshape((10, 2)).astype(float)
        dsource1 = TensorDataSource("ds1", ds1)
        dsource2 = TensorDataSource("ds2", np.arange(70).reshape((10, 7)).astype(float))
        dsource3 = TensorDataSource("ds3", np.arange(60).reshape((10, 3, 2)).astype(float))
        dsources = {ds.name: ds for ds in [dsource1, dsource2, dsource3]}
        encoders = {
            "ds1": [ScaleEncoder(scaler=StandardScaler())],
            "ds2": [ScaleEncoder(scaler=StandardScaler())],
            "ds3": [ScaleEncoder(scaler=StandardScaler())],
        }
        ds_test = DataSet(input_sources=dsources)
        splitter = OrderedSplitter(reverse=True)
        train, val, test = ds_test.split_encode(splitter, encoders=encoders)

        ds1_train_orig = np.array(train.input_sources["ds1"].decode())
        np.testing.assert_array_almost_equal(ds1_train_orig, ds1[4:])
        ds1_val_orig = np.array(val.input_sources["ds1"].decode())
        np.testing.assert_array_almost_equal(ds1_val_orig, ds1[2:4])
        ds1_test_orig = np.array(test.input_sources["ds1"].decode())
        np.testing.assert_array_almost_equal(ds1_test_orig, ds1[:2])
        ds2_train_orig = np.array(train.input_sources["ds2"].decode())
        np.testing.assert_array_almost_equal(ds2_train_orig, dsource2.tensors[4:])
        ds2_val_orig = np.array(val.input_sources["ds2"].decode())
        np.testing.assert_array_almost_equal(ds2_val_orig, dsource2.tensors[2:4])
        ds2_test_orig = np.array(test.input_sources["ds2"].decode())
        np.testing.assert_array_almost_equal(ds2_test_orig, dsource2.tensors[:2])
        ds3_train_orig = np.array(train.input_sources["ds3"].decode())
        np.testing.assert_array_almost_equal(ds3_train_orig, dsource3.tensors[4:])
        ds3_val_orig = np.array(val.input_sources["ds3"].decode())
        np.testing.assert_array_almost_equal(ds3_val_orig, dsource3.tensors[2:4])
        ds3_test_orig = np.array(test.input_sources["ds3"].decode())
        np.testing.assert_array_almost_equal(ds3_test_orig, dsource3.tensors[:2])


class TestTensorDataSource(TestCase):
    def test_fit_encode(self):
        na = np.array(
            [
                [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
                [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0], [12.0, 13.0]],
            ]
        )
        tds = TensorDataSource("test1", na)
        encoded = tds.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        true_encoded = encoded[0]
        expected_encoded = np.array([[0.0, 0.0], [0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]])
        np.testing.assert_array_almost_equal(true_encoded, expected_encoded)
        decoded = encoded.decode()
        np.testing.assert_array_almost_equal(decoded[0], na[0])

    def test_select_features(self):
        rows, cols = 5, 3
        na = np.arange(rows * cols).reshape((rows, cols))
        tds = TensorDataSource("test1", na)
        selected = tds.select_features([0, 2])
        self.assertEqual((rows, 2), selected.tensors.shape)
        expected_selected_na = na[:, [0, 2]]
        self.assertTrue(np.array_equal(expected_selected_na, selected[:]))

    def test_select_features_with_encoders(self):
        rows, cols = 5, 3
        na = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        tds = TensorDataSource("test1", na)
        encoded_tds = tds.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        selected = encoded_tds.select_features([0, 2])
        self.assertEqual((rows, 2), selected[:].shape)
        decoded = selected.decode()
        expected_na = na[:, [0, 2]]
        np.testing.assert_array_almost_equal(expected_na, decoded[:])


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
        expected_train = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        self.assertTrue(np.array_equal(train_ds[0], expected_train))
        expected_last_train = np.array([[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]])
        self.assertTrue(np.array_equal(train_ds[-1], expected_last_train))
        expected_val = np.array([[12, 13], [14, 15], [16, 17], [18, 19], [20, 21]])
        self.assertTrue(np.array_equal(val_ds[0], expected_val))
        expected_test = np.array([[16, 17], [18, 19], [20, 21], [22, 23], [24, 25]])
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
        self.assertTrue(np.array_equal(train_ds[0], na[: sr * l : sr]))

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
            [
                [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
                [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0], [12.0, 13.0]],
            ]
        )
        slice = tds[1:3]
        self.assertTrue(np.array_equal(slice, expected_slice))

    def test_encode(self):
        na = np.arange(20).reshape((10, 2))
        na[:, 1] = 0
        tds = TimeseriesDataSource("test1", na, length=5)
        encoded = tds.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])

        expected_encoded = np.array(
            [
                [0.0, 0.0],
                [0.11111111, 0.0],
                [0.22222222, 0.0],
                [0.33333333, 0.0],
                [0.44444444, 0.0],
            ],
        )
        true_encoded = encoded[0]
        np.testing.assert_array_almost_equal(true_encoded, expected_encoded)
        decoded = encoded.decode()
        np.testing.assert_array_almost_equal(decoded[0], na[:5])

    def test_get_targets(self):
        rows, cols = 15, 3
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 1] = 0
        na[:, 0] = np.arange(rows)
        target_params = TimeseriesTargetsParams(delay=1, pred_len=3, stride=2, target_idx=2)
        tds = TimeseriesDataSource("test1", na, length=5, stride=2, target_params=target_params)
        targets_ds = tds.get_targets()
        self.assertEqual(targets_ds.tensors.shape, (3, 3))
        expected_targets = np.array([[20, 26, 32], [26, 32, 38], [32, 38, 44]])
        self.assertTrue(np.array_equal(targets_ds[:], expected_targets))

    def test_split_with_targets_as_orig(self):
        splitter = OrderedSplitter(train=0.6, val=0.2)
        rows, cols = 20, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 0] = np.arange(rows)
        target_params = TimeseriesTargetsParams(delay=1, pred_len=3, stride=2)
        tds = TimeseriesDataSource(
            "test1",
            na,
            length=5,
            stride=1,
            sampling_rate=1,
            target_params=target_params,
        )
        self.assertEqual(len(tds), 10)
        train_ds, val_ds, test_ds = tds.split(splitter)
        self.assertEqual(len(train_ds), 6)
        self.assertEqual(len(val_ds), 2)
        self.assertEqual(len(test_ds), 2)
        expected_train = np.array([[0, 1], [1, 3], [2, 5], [3, 7], [4, 9]])
        self.assertTrue(np.array_equal(train_ds[0], expected_train))
        expected_last_train = np.array([[5, 11], [6, 13], [7, 15], [8, 17], [9, 19]])
        self.assertTrue(np.array_equal(train_ds[-1], expected_last_train))
        expected_val = np.array([[6, 13], [7, 15], [8, 17], [9, 19], [10, 21]])
        self.assertTrue(np.array_equal(val_ds[0], expected_val))
        expected_test = np.array([[8, 17], [9, 19], [10, 21], [11, 23], [12, 25]])
        self.assertTrue(np.array_equal(test_ds[0], expected_test))

    def test_select_features(self):
        rows, cols = 10, 3
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 0] = np.arange(rows)
        tds = TimeseriesDataSource("test1", na, length=5, stride=1, sampling_rate=1)
        selected = tds.select_features([0, 2])
        expected_first_element = np.array([[0, 2], [1, 5], [2, 8], [3, 11], [4, 14]])
        self.assertTrue(np.array_equal(selected[0], expected_first_element))

    def test_select_features_with_encoders(self):
        rows, cols = 10, 3
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 0] = np.arange(rows)
        tds = TimeseriesDataSource("test1", na, length=5, stride=1, sampling_rate=1)
        encoded = tds.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        selected = encoded.select_features([0, 2])
        decoded = selected.decode()
        expected_first_element = na[:5, [0, 2]]
        self.assertTrue(np.array_equal(decoded[0], expected_first_element))


class TestTargetTimeseriesDataSource(TestCase):
    def test_from_timeseries_nominal(self):
        rows, cols = 15, 3
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 1] = np.arange(rows)
        target_params = TimeseriesTargetsParams(delay=1, pred_len=3, stride=2, target_idx=1)
        tds = TimeseriesDataSource(
            "test1",
            na,
            length=5,
            stride=1,
            sampling_rate=1,
            target_params=target_params,
        )
        ttds = TargetTimeseriesDataSource.from_timeseries_datasource(tds, name="target1")
        self.assertEqual(len(tds), len(ttds))
        all_targets = ttds[:]
        expected_all_targets = np.array(
            [
                [6.0, 8.0, 10.0],
                [7.0, 9.0, 11.0],
                [8.0, 10.0, 12.0],
                [9.0, 11.0, 13.0],
                [10.0, 12.0, 14.0],
            ]
        )
        self.assertTrue(np.array_equal(all_targets, expected_all_targets))

    def test_from_timeseries_with_reverse(self):
        rows, cols = 15, 3
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 1] = np.arange(rows)
        target_params = TimeseriesTargetsParams(delay=1, pred_len=3, stride=2, target_idx=1, reverse=True)
        tds = TimeseriesDataSource(
            "test1",
            na,
            length=5,
            stride=1,
            sampling_rate=1,
            target_params=target_params,
        )
        ttds = TargetTimeseriesDataSource.from_timeseries_datasource(tds, name="target1")
        self.assertEqual(len(tds), len(ttds))
        all_targets = ttds[:]
        expected_all_targets = np.array(
            [
                [10.0, 8.0, 6.0],
                [11.0, 9.0, 7.0],
                [12.0, 10.0, 8.0],
                [13.0, 11.0, 9.0],
                [14.0, 12.0, 10.0],
            ]
        )
        self.assertTrue(np.array_equal(all_targets, expected_all_targets))

    def test_from_timeseries_with_encoders(self):
        rows, cols = 15, 3
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 1] = np.arange(rows)
        target_params = TimeseriesTargetsParams(delay=1, pred_len=3, stride=2, target_idx=1, reverse=True)
        tds = TimeseriesDataSource(
            "test1",
            na,
            length=5,
            stride=1,
            sampling_rate=1,
            target_params=target_params,
        )
        encoded = tds.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        ttds = TargetTimeseriesDataSource.from_timeseries_datasource(encoded, name="target1")
        decoded = ttds.decode()
        self.assertEqual(len(tds), len(decoded))
        all_targets = decoded[:]
        expected_all_targets = np.array(
            [
                [10.0, 8.0, 6.0],
                [11.0, 9.0, 7.0],
                [12.0, 10.0, 8.0],
                [13.0, 11.0, 9.0],
                [14.0, 12.0, 10.0],
            ]
        )
        self.assertTrue(np.array_equal(all_targets, expected_all_targets))

    def test_decode_predictions(self):
        rows, cols = 16, 3
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 1] = np.arange(rows)
        target_params = TimeseriesTargetsParams(delay=1, pred_len=3, stride=2, target_idx=1, reverse=True)
        tds = TimeseriesDataSource(
            "test1",
            na,
            length=5,
            stride=1,
            sampling_rate=1,
            target_params=target_params,
        )
        encoded = tds.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        ttds = TargetTimeseriesDataSource.from_timeseries_datasource(encoded, name="target1")
        predictions = np.array([[0.53333333, 0.4, 0.66666667], [0.86666667, 0.73333333, 1.0]])
        encoded_tensors = TensorDataSource("decoder", predictions, encoders=ttds.get_encoders())
        decoded_tensors = encoded_tensors.decode()

        expected_decoded = np.array([[8.0, 6.0, 10.0], [13.0, 11.0, 15.0]])
        np.testing.assert_array_almost_equal(expected_decoded, decoded_tensors[:])

    def test_split_with_targets_as_orig(self):
        splitter = OrderedSplitter(train=0.6, val=0.2)
        rows, cols = 20, 2
        na = np.arange(rows * cols).reshape((rows, cols))
        na[:, 0] = np.arange(rows)
        target_params = TimeseriesTargetsParams(delay=1, pred_len=3, stride=2)
        tds = TimeseriesDataSource(
            "test1",
            na,
            length=5,
            stride=1,
            sampling_rate=1,
            target_params=target_params,
        )
        ttds = TargetTimeseriesDataSource.from_timeseries_datasource(tds, name="target1")
        self.assertEqual(len(tds), len(ttds))
        train_ds, val_ds, test_ds = tds.split(splitter)
        train_tds, val_tds, test_tds = ttds.split(splitter)
        self.assertEqual(len(train_ds), len(train_tds))
        self.assertEqual(len(val_ds), len(val_tds))
        self.assertEqual(len(test_ds), len(test_tds))
        np.testing.assert_array_equal(ttds[:6], train_tds[:])
        np.testing.assert_array_equal(ttds[6:8], val_tds[:])
        np.testing.assert_array_equal(ttds[8:], test_tds[:])


class TestCompoundDataSource(TestCase):
    def test_nominal(self):
        rows, cols = 5, 3
        na1 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 += 0.5
        tds1 = TensorDataSource("test1", na1)
        tds2 = TensorDataSource("test2", na2)
        tds_comp = tds1 + tds2
        self.assertEqual(len(tds_comp), len(tds1))
        i1, i2 = tds1[0], tds2[0]
        i_comp = np.concatenate((i1, i2))
        self.assertTrue(np.array_equal(tds_comp[0], i_comp))

    def test_with_timeseries(self):
        rows, cols = 10, 3
        na1 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 += 0.5
        tds1 = TimeseriesDataSource("test1", na1, length=5)
        tds2 = TimeseriesDataSource("test2", na2, length=5)
        tds_comp = tds1 + tds2
        self.assertEqual(len(tds_comp), len(tds1))
        expected_el = np.array(
            [
                [0.0, 1.0, 2.0, 0.5, 1.5, 2.5],
                [3.0, 4.0, 5.0, 3.5, 4.5, 5.5],
                [6.0, 7.0, 8.0, 6.5, 7.5, 8.5],
                [9.0, 10.0, 11.0, 9.5, 10.5, 11.5],
                [12.0, 13.0, 14.0, 12.5, 13.5, 14.5],
            ]
        )
        self.assertTrue(np.array_equal(expected_el, tds_comp[0]))

    def test_as_numpy(self):
        rows, cols = 5, 2
        na1 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 += 0.5
        tds1 = TimeseriesDataSource("test1", na1, length=3)
        tds2 = TimeseriesDataSource("test2", na2, length=3)
        tds_comp = tds1 + tds2
        self.assertEqual(len(tds_comp), len(tds1))
        expected_el = np.array(
            [
                [0.0, 1.0, 0.5, 1.5],
                [2.0, 3.0, 2.5, 3.5],
                [4.0, 5.0, 4.5, 5.5],
                [6.0, 7.0, 6.5, 7.5],
                [8.0, 9.0, 8.5, 9.5],
            ]
        )
        self.assertTrue(np.array_equal(expected_el, tds_comp.as_numpy()))

    def test_encode_decode_tensor_ds(self):
        rows, cols = 5, 2
        na1 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 += 0.5
        na3 = np.arange(rows * 3, dtype=np.float64).reshape((rows, 3))
        tds1 = TensorDataSource("test1", na1)
        tds2 = TensorDataSource("test2", na2)
        tds3 = TensorDataSource("test3", na3)
        tds_comp: TensorDataSource = tds1 + tds2 + tds3
        encoded = tds_comp.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        tds1_encoded = tds1.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        tds2_encoded = tds2.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        tds3_encoded = tds3.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        expected_el = np.concatenate((tds1_encoded[0], tds2_encoded[0], tds3_encoded[0]), axis=-1)
        np.testing.assert_array_almost_equal(expected_el, encoded[0])
        tds_comp_decoded = encoded.decode()
        np.testing.assert_array_almost_equal(tds_comp[:], tds_comp_decoded[:])

        encoders = encoded.get_encoders()
        re_encoded_tds = tds_comp.encode(encoders)
        np.testing.assert_array_almost_equal(expected_el, re_encoded_tds[0])

        re_decoded_tds = re_encoded_tds.decode()
        np.testing.assert_array_almost_equal(tds_comp[:], re_decoded_tds[:])

    def test_encode_decode_timeseries(self):
        rows, cols = 5, 2
        na1 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 += 0.5
        na3 = np.arange(rows * 3, dtype=np.float64).reshape((rows, 3))
        tds1 = TimeseriesDataSource("test1", na1, length=3)
        tds2 = TimeseriesDataSource("test2", na2, length=3)
        tds3 = TimeseriesDataSource("test3", na3, length=3)
        tds_comp: TimeseriesDataSource = tds1 + tds2
        tds_comp = tds_comp + tds3
        encoded = tds_comp.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        tds1_encoded = tds1.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        tds2_encoded = tds2.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        tds3_encoded = tds3.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        expected_el = np.concatenate((tds1_encoded[0], tds2_encoded[0], tds3_encoded[0]), axis=-1)
        np.testing.assert_array_almost_equal(expected_el, encoded[0])
        tds_comp_decoded = encoded.decode()
        np.testing.assert_array_almost_equal(tds_comp[:], tds_comp_decoded[:])

        encoders = encoded.get_encoders()
        re_encoded_tds = tds_comp.encode(encoders)
        np.testing.assert_array_almost_equal(expected_el, re_encoded_tds[0])

        re_decoded_tds = re_encoded_tds.decode()
        np.testing.assert_array_almost_equal(tds_comp[:], re_decoded_tds[:])

    def test_decode_data_timeseries(self):
        rows, cols = 5, 2
        na1 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 += 0.5
        tds1 = TimeseriesDataSource("test1", na1, length=3)
        tds2 = TimeseriesDataSource("test2", na2, length=3)
        tds_comp: CompoundDataSource = tds1 + tds2
        encoded = tds_comp.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        data_to_decode = encoded[:]
        ds_to_decode = TensorDataSource("test_decode", data_to_decode, encoders=encoded.get_encoders())
        decoded_data_ds = ds_to_decode.decode()
        np.testing.assert_array_almost_equal(tds_comp[:], decoded_data_ds[:])

    def test_decode_data_tensors(self):
        rows, cols = 5, 2
        na1 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        na2 += 0.5
        tds1 = TensorDataSource("test1", na1)
        tds2 = TensorDataSource("test2", na2)
        tds_comp: CompoundDataSource = tds1 + tds2
        encoded = tds_comp.fit_encode(encoders=[ScaleEncoder(MinMaxScaler())])
        data_to_decode = encoded[:]
        ds_to_decode = TensorDataSource("test_decode", data_to_decode, encoders=encoded.get_encoders())
        decoded_data_ds = ds_to_decode.decode()
        np.testing.assert_array_almost_equal(tds_comp[:], decoded_data_ds[:])


class TestXYBatchGenerator(TestCase):
    def test_on_epoch_end(self):
        rows, cols = 10, 2
        na = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        dsource1 = TensorDataSource("ds1", na)
        na2 = np.arange(rows * 3, dtype=np.float64).reshape((rows, 3))
        dsource2 = TensorDataSource("ds2", na2)
        na3 = np.arange(rows * 2, dtype=np.float64).reshape((rows, 2))
        dsource3 = TensorDataSource("ds3", na3)
        dsources = {ds.name: ds for ds in [dsource1, dsource2]}
        encoders = {
            dsource1.name: [ScaleEncoder(scaler=MinMaxScaler())],
            dsource2.name: [ScaleEncoder(scaler=MinMaxScaler())],
            dsource3.name: [ScaleEncoder(scaler=MinMaxScaler())],
        }
        ds_test = DataSet(input_sources=dsources, target_sources={dsource3.name: dsource3})
        splitter = OrderedSplitter()
        train, val, test = ds_test.split_encode(splitter, encoders=encoders)
        xy_gen = XYBatchGenerator(train.input_sources, train.target_sources, batch_size=2, shuffle=False)
        batch = xy_gen[np.int32(0)]
        expected_batch = (
            {
                "ds1": array([[0.0, 0.0], [0.2, 0.2]]),
                "ds2": array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]]),
            },
            {"ds3": array([[0.0, 0.0], [0.2, 0.2]])},
        )
        np.testing.assert_array_almost_equal(expected_batch[0]["ds1"], batch[0]["ds1"])
        np.testing.assert_array_almost_equal(expected_batch[0]["ds2"], batch[0]["ds2"])
        np.testing.assert_array_almost_equal(expected_batch[1]["ds3"], batch[1]["ds3"])


class TestXYWBatchGenerator(TestCase):
    def test_on_epoch_end(self):
        rows, cols = 10, 2
        na = np.arange(rows * cols, dtype=np.float64).reshape((rows, cols))
        dsource1 = TensorDataSource("ds1", na)
        na2 = np.arange(rows * 3, dtype=np.float64).reshape((rows, 3))
        dsource2 = TensorDataSource("ds2", na2)
        na3 = np.arange(rows * 2, dtype=np.float64).reshape((rows, 2))
        dsource3 = TensorDataSource("ds3", na3)
        dsources = {ds.name: ds for ds in [dsource1, dsource2]}
        encoders = {
            dsource1.name: [ScaleEncoder(scaler=MinMaxScaler())],
            dsource2.name: [ScaleEncoder(scaler=MinMaxScaler())],
            dsource3.name: [ScaleEncoder(scaler=MinMaxScaler())],
        }
        ds_test = DataSet(input_sources=dsources, target_sources={dsource3.name: dsource3})
        sample_weights = np.arange(rows)
        weights = {"weights": TensorDataSource("weights", tensors=sample_weights)}

        splitter = OrderedSplitter()
        train, val, test = ds_test.split_encode(splitter, encoders=encoders)
        xyw_gen = XYWBatchGenerator(train.input_sources, train.target_sources, weights, batch_size=2, shuffle=False)
        batch = xyw_gen[np.int32(0)]
        expected_batch = (
            {
                "ds1": array([[0.0, 0.0], [0.2, 0.2]]),
                "ds2": array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]]),
            },
            {"ds3": array([[0.0, 0.0], [0.2, 0.2]])},
        )
        np.testing.assert_array_almost_equal(expected_batch[0]["ds1"], batch[0]["ds1"])
        np.testing.assert_array_almost_equal(expected_batch[0]["ds2"], batch[0]["ds2"])
        np.testing.assert_array_almost_equal(expected_batch[1]["ds3"], batch[1]["ds3"])
        np.testing.assert_array_almost_equal(batch[2]["weights"], sample_weights[0:2])


if __name__ == "__main__":
    main()

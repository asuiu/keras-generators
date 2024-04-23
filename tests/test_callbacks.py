import inspect
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
from tf_keras import Model, Sequential
from tf_keras.src.layers import Dense

from keras_generators.callbacks import (
    EarlyStoppingAtMinLoss,
    MetricCheckpoint,
    SerializableCSVLogger,
    SerializableReduceLROnPlateau,
    SerializableTensorBoard,
)


def _is_function_object(obj) -> bool:
    # Check if v is a lambda function
    if isinstance(obj, type(lambda: 0)) and obj.__name__ == "<lambda>":
        return True
    if inspect.ismethod(obj):
        return True
    return False


class TestSerializableKerasObject(TestCase):
    def setUp(self):
        self.model = Sequential()
        self.model.add(Dense(2, input_shape=(2,)))
        self.model.compile(optimizer="adam", loss="mse")

    @staticmethod
    def models_equal(m1: Model, m2: Model, tol=1e-5) -> bool:
        weights1 = m1.get_weights()
        weights2 = m2.get_weights()
        for w1, w2 in zip(weights1, weights2):
            if not np.allclose(w1, w2, atol=tol):
                return False

        json_eq = m1.to_json() == m2.to_json()
        if not json_eq:
            return False
        return True

    def test_serialize_deserialize_EarlyStoppingAtMinLoss(self):
        # set up the callback
        loss_max_diff = 0.01
        patience = 10
        min_delta = 1e-4

        callback = EarlyStoppingAtMinLoss(loss_max_diff=loss_max_diff, patience=patience, min_delta=min_delta)
        callback.set_model(self.model)
        callback.on_train_begin()
        callback.on_epoch_begin(0)
        callback.on_batch_begin(0)
        callback.on_batch_end(0)
        callback.on_epoch_end(0, logs={"val_loss": None})

        # Serialize and deserialize the callback
        serialized = callback.serialize()
        deserialized = EarlyStoppingAtMinLoss.deserialize(serialized)
        for k, v in callback.__dict__.items():
            if _is_function_object(v):
                continue
            deserialized_v = getattr(deserialized, k)
            if isinstance(v, Model):
                self.assertTrue(self.models_equal(v, deserialized_v))
                continue
            self.assertEqual(v, deserialized_v)

        deserialized.on_epoch_begin(1)
        deserialized.on_batch_begin(1)
        deserialized.on_batch_end(1)
        deserialized.on_epoch_end(1, logs={"val_loss": None})
        deserialized.on_train_end()

    def test_serialize_deserialize_MetricCheckpoint(self):
        # set up the callback
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            callback = MetricCheckpoint(model_dir)
            callback.set_model(self.model)
            callback.on_train_begin()
            callback.on_epoch_begin(0)
            callback.on_batch_begin(0)
            callback.on_batch_end(0)
            callback.on_epoch_end(0, logs={"val_loss": None})

            # Serialize and deserialize the callback
            serialized = callback.serialize()
            deserialized = MetricCheckpoint.deserialize(serialized)
            for k, v in callback.__dict__.items():
                if _is_function_object(v):
                    continue
                deserialized_v = getattr(deserialized, k)
                if k in ("_f", "_je"):
                    continue
                if isinstance(v, Model):
                    self.assertTrue(self.models_equal(v, deserialized_v))
                    continue
                self.assertEqual(v, deserialized_v, f"Failed on {k} with {v} != {deserialized_v}")

            deserialized.on_epoch_begin(1)
            deserialized.on_batch_begin(1)
            deserialized.on_batch_end(1)
            deserialized.on_epoch_end(1, logs={"val_loss": None})
            deserialized.on_train_end()
            callback.on_train_end()
            callback._f.close()
        print()

    def test_serialize_deserialize_SerializableReduceLROnPlateau(self):
        # set up the callback
        monitor = "val_loss"
        factor = 0.1
        patience = 10
        verbose = 0
        mode = "auto"
        min_delta = 1e-4
        cooldown = 0
        min_lr = 0

        callback = SerializableReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
        )
        callback.set_model(self.model)
        callback.on_train_begin()
        callback.on_epoch_begin(0)
        callback.on_batch_begin(0)
        callback.on_batch_end(0)
        callback.on_epoch_end(0)

        # Serialize and deserialize the callback
        serialized = callback.serialize()
        deserialized = SerializableReduceLROnPlateau.deserialize(serialized)
        for k, v in callback.__dict__.items():
            if _is_function_object(v):
                continue
            deserialized_v = getattr(deserialized, k)
            if isinstance(v, Model):
                self.assertTrue(self.models_equal(v, deserialized_v))
                continue
            self.assertEqual(v, deserialized_v)

        deserialized.on_epoch_begin(1)
        deserialized.on_batch_begin(1)
        deserialized.on_batch_end(1)
        deserialized.on_epoch_end(1)
        deserialized.on_train_end()

    def test_serialize_deserialize_SerializableTensorBoard(self):
        callback = SerializableTensorBoard(
            log_dir="logs",
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )
        callback.set_model(self.model)
        callback.on_train_begin()
        callback.on_epoch_begin(0)
        callback.on_batch_begin(0)
        callback.on_batch_end(0)
        callback.on_epoch_end(0)

        serialized = callback.serialize()
        deserialized = SerializableTensorBoard.deserialize(serialized)
        for k, v in callback.__dict__.items():
            if _is_function_object(v):
                continue
            if k in ("_writers",):
                continue
            deserialized_v = getattr(deserialized, k)
            if isinstance(v, Model):
                self.assertTrue(self.models_equal(v, deserialized_v))
                continue
            self.assertEqual(v, deserialized_v)
        deserialized.on_epoch_begin(1)
        deserialized.on_batch_begin(1)
        deserialized.on_batch_end(1)
        deserialized.on_epoch_end(1)
        deserialized.on_train_end()

    def test_serialize_deserialize_SerializableCSVLogger(self):
        # set up the callback
        callback = SerializableCSVLogger(filename="logs.csv", separator=";", append=True)
        callback.set_model(self.model)
        callback.on_train_begin()
        callback.on_epoch_begin(0)
        callback.on_batch_begin(0)
        callback.on_batch_end(0)
        callback.on_epoch_end(0)

        # Serialize and deserialize the callback
        serialized = callback.serialize()
        deserialized = SerializableCSVLogger.deserialize(serialized)

        # assert that the rest of the callback functions still work
        for k, v in callback.__dict__.items():
            if _is_function_object(v):
                continue
            if k in ("writer", "csv_file"):
                continue
            deserialized_v = getattr(deserialized, k)
            if isinstance(v, Model):
                self.assertTrue(self.models_equal(v, deserialized_v))
                continue
            self.assertEqual(v, deserialized_v)

        deserialized.on_train_begin()
        deserialized.on_epoch_begin(1)
        deserialized.on_batch_begin(1)
        deserialized.on_batch_end(1)
        deserialized.on_epoch_end(1)
        deserialized.on_train_end()

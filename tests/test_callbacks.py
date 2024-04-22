import inspect
from unittest import TestCase

from keras_generators.callbacks import (
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


class TestSerializableReduceLROnPlateau(TestCase):
    def test_serialize_deserialize(self):
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
        serialized = callback.serialize()
        deserialized = SerializableReduceLROnPlateau.deserialize(serialized)
        for k, v in callback.__dict__.items():
            if _is_function_object(v):
                continue
            self.assertEqual(v, getattr(deserialized, k))


class TestSerializableTensorBoard(TestCase):
    def test_serialize_deserialize(self):
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
        serialized = callback.serialize()
        deserialized = SerializableTensorBoard.deserialize(serialized)
        self.assertDictEqual(deserialized.__dict__, callback.__dict__)


class TestSerializableCSVLogger(TestCase):
    def test_serialize_deserialize(self):
        callback = SerializableCSVLogger(filename="logs.csv", separator=";", append=True)
        serialized = callback.serialize()
        deserialized = SerializableCSVLogger.deserialize(serialized)
        self.assertDictEqual(deserialized.__dict__, callback.__dict__)

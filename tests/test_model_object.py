from unittest import TestCase
from unittest.mock import MagicMock, patch

from keras_generators.generators import XYBatchGenerator, XBatchGenerator, DataSource

from keras_generators.model_abstractions.model_object import ModelObject


class TestModelObject(TestCase):
    def setUp(self):
        mp = MagicMock()
        model = MagicMock()
        encoders = MagicMock()
        auto_clear = 2
        self.model_object = ModelObject(mp, model, encoders, auto_clear)

    @patch('keras.backend.clear_session')
    def test_clear_session_on_prediction(self, mock_clear_session):
        x_batch_generator = MagicMock(spec=XBatchGenerator)

        self.model_object.predict(x_batch_generator)
        mock_clear_session.assert_not_called()
        self.assertEqual(self.model_object._state_calls, 1)

        self.model_object.predict(x_batch_generator)
        mock_clear_session.assert_called_once()
        self.assertEqual(self.model_object._state_calls, 0)

    @patch('keras.backend.clear_session')
    def test_clear_session_on_evaluation(self, mock_clear_session):
        data_source = MagicMock(spec=DataSource)
        data_source.encode.return_value = data_source
        x_y_batch_generator = MagicMock(spec=XYBatchGenerator)
        x_y_batch_generator.inputs = {"data_source": data_source}
        x_y_batch_generator.targets = {"data_source": data_source}
        x_y_batch_generator.batch_size = 1

        self.model_object.evaluate_raw(x_y_batch_generator)
        mock_clear_session.assert_not_called()
        self.assertEqual(self.model_object._state_calls, 1)

        self.model_object._state_calls = self.model_object._state_autoclear
        self.model_object.evaluate_raw(x_y_batch_generator)
        mock_clear_session.assert_called_once()
        self.assertEqual(self.model_object._state_calls, 0)


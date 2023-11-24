from unittest import TestCase
from unittest.mock import MagicMock, patch

from keras_generators.model_abstractions.model_object import ModelObject


class TestModelObject(TestCase):
    def setUp(self):
        mp = MagicMock()
        model = MagicMock()
        encoders = MagicMock()
        self.model_object = ModelObject(mp, model, encoders)

    @patch('keras.backend.clear_session')
    def test_check_clear_state(self, mock_clear_session):
        self.model_object._state_calls = 0
        self.model_object._check_clear_state()
        self.assertEqual(self.model_object._state_calls, 1)
        self.model_object._check_clear_state()
        self.assertEqual(self.model_object._state_calls, 2)
        self.model_object._state_calls = self.model_object._state_autoclear
        self.model_object._check_clear_state()
        mock_clear_session.assert_called_once()
        self.assertEqual(self.model_object._state_calls, 0)


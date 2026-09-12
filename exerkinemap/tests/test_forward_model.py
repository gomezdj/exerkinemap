# test_forward_model.py
import pytest
from workflows.forward_model import ForwardModel

def test_forward_prediction():
    model = ForwardModel()
    prediction = model.predict(sequence_embedding=[0.1, 0.5, 0.2])
    assert prediction.shape[0] == 1 # Expected output dimension
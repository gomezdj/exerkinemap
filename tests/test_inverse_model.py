# test_inverse_model.py
import pytest
from workflows.inverse_model import InverseModel

def test_inverse_sequence_design():
    model = InverseModel()
    designed_sequence = model.design(target_profile=[0.9, 0.1, 0.0])
    assert isinstance(designed_sequence, str)
    assert len(designed_sequence) > 0
import os

def test_model_exists():
    assert os.path.exists("model.pkl"), "Model file not found!"

import pytest
import torch
from src.modules.dataset_module import LatLongDataset

@pytest.fixture
def dataset():
    return LatLongDataset('datasets/test_dataset.csv', sequence_length=4)

def test_len(dataset):
    assert len(dataset) == len(dataset.data) - dataset.sequence_lenght

def test_getitem(dataset):
    sample_index = 0
    sample_sequence, sample_target = dataset[sample_index]
    assert sample_sequence.shape == (dataset.sequence_lenght, 2)
    assert sample_target.shape == (1, 2)

def test_sequence_values(dataset):
    sample_index = 0
    sample_sequence, _ = dataset[sample_index]
    max_longitude = 180
    min_longitude = -180
    max_latitude = 90
    min_latitude = -90
    assert (sample_sequence >= min_longitude).all() and (sample_sequence <= max_longitude).all()
    assert (sample_sequence >= min_latitude).all() and (sample_sequence <= max_latitude).all()

def test_sequence_normalization(dataset):
    sample_index = 0
    sample_sequence, _ = dataset[sample_index]
    max_value = 1.0
    min_value = -1.0
    assert (sample_sequence >= min_value).all() and (sample_sequence <= max_value).all()

def test_target_normalization(dataset):
    sample_index = 0
    _, sample_target = dataset[sample_index]
    max_value = 1.0
    min_value = -1.0
    assert (sample_target >= min_value).all() and (sample_target <= max_value).all()

def test_target(dataset):
    sample_index = 0
    _, sample_target = dataset[sample_index]
    original_index = sample_index + dataset.sequence_lenght
    original_data = dataset.data.iloc[original_index].values
    original_target = original_data / 180
    assert torch.allclose(sample_target, torch.tensor(original_target, dtype=torch.float32))
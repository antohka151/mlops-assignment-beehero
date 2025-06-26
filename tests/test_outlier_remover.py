import pandas as pd
import pytest
from src.data.outlier_remover.component import OutlierRemover
from src.data.outlier_remover.schema import OutlierRemoverConfig

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'sensor_id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C'],
        'temperature_sensor': [20, 22, 21, 55, 54, 12, 11, 13, 12],
        'other_col': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'size': [1, 1, 1, 0, 0, 1, 1, 1, 1],
    })
    return data

@pytest.fixture
def config():
    return OutlierRemoverConfig(
        group_by_column='sensor_id',
        temperature_column='temperature_sensor',
        max_temp_threshold=50,
        min_temp_threshold=10
    )

def test_fit_stores_group_means(sample_data, config):
    remover = OutlierRemover(config)
    remover.fit(sample_data)
    means = remover._group_means
    assert means['A'] == pytest.approx(21)
    assert means['B'] == pytest.approx(54.5)
    assert means['C'] == pytest.approx(12)

def test_transform_removes_outlier_groups(sample_data, config):
    remover = OutlierRemover(config)
    remover.fit(sample_data)
    X_filtered, _ = remover.transform(sample_data)
    # Group B should be removed (mean=54.5 > 50)
    assert 'B' not in X_filtered['sensor_id'].values
    # Groups A and C should remain
    assert set(X_filtered['sensor_id'].unique()) == {'A', 'C'}
    # Row count matches only A and C rows
    assert len(X_filtered) == 7

def test_transform_with_y(sample_data, config):
    remover = OutlierRemover(config)
    remover.fit(sample_data)
    y = sample_data['size']
    X_filtered, y_filtered = remover.transform(sample_data, y)
    assert len(X_filtered) == len(y_filtered)
    assert all(X_filtered.index == y_filtered.index)

def test_fit_transform_equivalence(sample_data, config):
    remover = OutlierRemover(config)
    X1, _ = remover.fit_transform(sample_data)
    remover2 = OutlierRemover(config)
    remover2.fit(sample_data)
    X2, _ = remover2.transform(sample_data)
    pd.testing.assert_frame_equal(X1, X2)

def test_transform_raises_if_not_fitted(sample_data, config):
    remover = OutlierRemover(config)
    with pytest.raises(Exception):
        remover.transform(sample_data)

def test_missing_columns_raises(sample_data, config):
    remover = OutlierRemover(config)
    bad_data = sample_data.drop(columns=['temperature_sensor'])
    with pytest.raises(ValueError):
        remover.fit(bad_data)

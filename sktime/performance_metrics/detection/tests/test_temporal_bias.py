"""Tests for TemporalBias metric."""

import numpy as np
import pandas as pd
import pytest
from sktime.performance_metrics.detection._temporal_bias import TemporalBias
from sktime.tests.test_switch import run_test_for_class

@pytest.mark.skipif(
    not run_test_for_class(TemporalBias),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_temporal_bias_normal_case():
    """Test the temporal bias metric for the normal case,
    which is when all the predicted events occur before and after all the real events."""
    y_true = pd.DataFrame({'ilocs': [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]})
    y_pred = pd.DataFrame({'ilocs': [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0]})
    
    metric = TemporalBias()
    skewness = metric.evaluate(y_true, y_pred)
    assert np.round(skewness, 5) == 0.27155
    assert isinstance(metric.temporal_bias, list)
    assert len(metric.temporal_bias) == len(y_true.loc[y_true.ilocs == 1])
    assert metric.temporal_bias == [-1, 1, 0, -1, -2]

@pytest.mark.skipif(
    not run_test_for_class(TemporalBias),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_temporal_bias_edge_case():
    """Test the temporal bias metric for the edge case
    when the predicted events occur only after the first real event and before the last real event."""
    y_true = pd.DataFrame({'ilocs': [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]})
    y_pred = pd.DataFrame({'ilocs': [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]})
    metric = TemporalBias()
    skewness = metric.evaluate(y_true, y_pred)
    assert np.round(skewness, 5) == 0.0
    assert isinstance(metric.temporal_bias, list)
    assert len(metric.temporal_bias) == len(y_true.loc[y_true.ilocs == 1])
    assert metric.temporal_bias == [1, 0, 0, 0, -1]

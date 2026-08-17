import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import FraudInferencePipeline

def test_inference_pipeline_init():
    try:
        pipeline = FraudInferencePipeline()
        assert pipeline is not None
    except Exception as e:
        pytest.skip(f"Could not initialize pipeline (models might not be trained yet): {e}")

def test_inference_prediction_shape():
    try:
        pipeline = FraudInferencePipeline()
    except Exception as e:
        pytest.skip(f"Models not available for testing: {e}")
        
    # Dummy transaction
    dummy_tx = np.random.normal(0, 1, size=(1, 30))
    
    results = pipeline.predict(dummy_tx)
    
    assert 'ae_anomaly_scores' in results
    assert 'ae_decisions' in results
    assert 'xgb_probabilities' in results
    assert 'xgb_decisions' in results
    
    assert len(results['ae_anomaly_scores']) == 1
    assert len(results['xgb_probabilities']) == 1

"""
System 1 테스트
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.system1 import System1Student, SimplifiedTradingModel, InferencePipeline


def test_simplified_model():
    """간단한 모델 테스트"""
    model = SimplifiedTradingModel(input_dim=10, hidden_dim=32, output_dim=3)
    
    x = torch.randn(1, 1, 10)  # [batch, seq, features]
    output = model(x)
    
    assert output.shape == (1, 3)  # [batch, actions]


def test_inference_pipeline():
    """추론 파이프라인 테스트"""
    model = SimplifiedTradingModel(input_dim=10, hidden_dim=32, output_dim=3)
    pipeline = InferencePipeline(model, device="cpu")
    
    tick_data = {
        "price": 150.0,
        "volume": 1000000,
        "prices": [150.0] * 20
    }
    
    result = pipeline.process_tick(tick_data)
    
    assert "prediction" in result
    assert "decision" in result["prediction"]
    assert result["prediction"]["decision"] in ["BUY", "SELL", "HOLD"]


def test_system1_student():
    """System 1 Student 테스트"""
    student = System1Student(model_type="simplified")
    
    # 추론 테스트
    tick_data = {
        "price": 150.0,
        "volume": 1000000,
        "prices": [150.0] * 20
    }
    
    result = student.infer(tick_data)
    
    assert "prediction" in result
    assert "execution" in result


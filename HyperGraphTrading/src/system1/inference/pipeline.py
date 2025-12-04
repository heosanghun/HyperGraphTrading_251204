"""
실시간 추론 파이프라인
"""
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

from ..model.architecture import LightweightTradingModel, SimplifiedTradingModel


class InferencePipeline:
    """추론 파이프라인"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 device: str = "cpu",
                 batch_size: int = 1):
        """파이프라인 초기화"""
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        
        # 성능 측정
        self.inference_times = []
    
    def preprocess(self, tick_data: Dict[str, Any]) -> torch.Tensor:
        """틱 데이터 전처리"""
        # 입력 특징 추출
        features = []
        
        # 가격 데이터
        if "price" in tick_data:
            features.append(tick_data["price"])
        if "volume" in tick_data:
            features.append(tick_data["volume"])
        if "high" in tick_data and "low" in tick_data:
            features.append(tick_data["high"] - tick_data["low"])  # 변동폭
        
        # 기술적 지표 (간단한 구현)
        if "prices" in tick_data and len(tick_data["prices"]) > 0:
            prices = tick_data["prices"][-20:]  # 최근 20개
            if len(prices) > 1:
                # 이동평균
                ma = np.mean(prices)
                features.append(ma)
                # 변동성
                volatility = np.std(prices)
                features.append(volatility)
        
        # 특징 벡터 생성 (패딩/자르기)
        while len(features) < 10:
            features.append(0.0)
        features = features[:10]
        
        # 정규화
        features = np.array(features, dtype=np.float32)
        if np.max(np.abs(features)) > 0:
            features = features / (np.max(np.abs(features)) + 1e-8)
        
        # 시계열 형태로 변환 (1 timestep)
        features = features.reshape(1, 1, -1)  # [batch, seq, features]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def infer(self, data: torch.Tensor) -> Dict[str, Any]:
        """추론 실행"""
        start_time = time.time()
        
        data = data.to(self.device)
        
        with torch.no_grad():
            if isinstance(self.model, (LightweightTradingModel, SimplifiedTradingModel)):
                output = self.model(data)
                if isinstance(output, tuple):
                    logits, value = output
                else:
                    logits = output
                    value = None
            else:
                logits = self.model(data)
                value = None
        
        # 확률 계산
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        
        # 결정 매핑
        decision_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        decision = decision_map.get(prediction, "HOLD")
        confidence = probs[0, prediction].item()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        result = {
            "decision": decision,
            "confidence": confidence,
            "probabilities": {
                "BUY": probs[0, 0].item(),
                "SELL": probs[0, 1].item(),
                "HOLD": probs[0, 2].item()
            },
            "inference_time_ms": inference_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if value is not None:
            result["value"] = value.item()
        
        return result
    
    def postprocess(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """후처리"""
        # 신뢰도 기반 필터링
        if prediction["confidence"] < 0.5:
            prediction["decision"] = "HOLD"
            prediction["filtered"] = True
        else:
            prediction["filtered"] = False
        
        return prediction
    
    def execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """의사결정 실행 (시뮬레이션)"""
        execution = {
            "action": decision["decision"],
            "confidence": decision["confidence"],
            "timestamp": decision["timestamp"],
            "executed": True,
            "simulated": True  # 실제 거래는 별도 모듈에서
        }
        
        return execution
    
    def process_tick(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """틱 데이터 처리 (전체 파이프라인)"""
        # 전처리
        processed_data = self.preprocess(tick_data)
        
        # 추론
        prediction = self.infer(processed_data)
        
        # 후처리
        final_decision = self.postprocess(prediction)
        
        # 실행 (시뮬레이션)
        execution = self.execute_decision(final_decision)
        
        return {
            "prediction": final_decision,
            "execution": execution
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            "mean_inference_time_ms": float(np.mean(times)),
            "median_inference_time_ms": float(np.median(times)),
            "p95_inference_time_ms": float(np.percentile(times, 95)),
            "p99_inference_time_ms": float(np.percentile(times, 99)),
            "min_inference_time_ms": float(np.min(times)),
            "max_inference_time_ms": float(np.max(times)),
            "total_inferences": len(times)
        }


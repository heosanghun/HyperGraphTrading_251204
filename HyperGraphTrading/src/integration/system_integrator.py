"""
System 2 ↔ System 1 통합 모듈
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch

from ..system2.system2_teacher import System2Teacher
from ..system1.system1_student import System1Student
from ..hypergraph import FinancialHypergraph


class SystemIntegrator:
    """시스템 통합 클래스"""
    
    def __init__(self,
                 hypergraph: FinancialHypergraph,
                 system2: System2Teacher,
                 system1: System1Student):
        """통합기 초기화"""
        self.hypergraph = hypergraph
        self.system2 = system2
        self.system1 = system1
        self.policy_history: List[Dict[str, Any]] = []
    
    def update_system1_from_system2(self,
                                   symbol: str,
                                   date: str,
                                   num_policies: int = 10) -> Dict[str, Any]:
        """System 2 정책으로 System 1 업데이트"""
        # System 2에서 정책 생성
        policy_result = self.system2.generate_policy(symbol, date, use_llm=False)
        policy = policy_result["policy"]
        
        # 정책 히스토리에 추가
        self.policy_history.append(policy)
        
        # 최근 정책들로 System 1 학습
        recent_policies = self.policy_history[-num_policies:]
        
        # 학습 데이터 준비 (간단한 구현)
        training_data = self._prepare_training_data(symbol, len(recent_policies))
        
        # 학습
        training_result = self.system1.train_from_teacher(
            teacher_policies=recent_policies,
            training_data=training_data,
            epochs=5,
            learning_rate=0.001
        )
        
        return {
            "policy": policy,
            "training_result": training_result
        }
    
    def _prepare_training_data(self, symbol: str, batch_size: int) -> torch.Tensor:
        """학습 데이터 준비"""
        # 실제로는 하이퍼그래프에서 특징 추출
        # 여기서는 간단한 랜덤 데이터
        return torch.randn(batch_size, 1, 10)  # [batch, seq, features]
    
    def run_realtime(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """실시간 실행 (System 1 사용)"""
        return self.system1.infer(tick_data)
    
    def should_update_from_system2(self,
                                  recent_decisions: List[Dict[str, Any]],
                                  threshold: float = 0.3) -> bool:
        """System 2 업데이트 필요 여부 판단"""
        if len(recent_decisions) < 5:
            return False
        
        # 최근 결정들의 신뢰도 확인
        confidences = [d.get("confidence", 0) for d in recent_decisions[-5:]]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 신뢰도가 낮으면 System 2 재학습 필요
        if avg_confidence < threshold:
            return True
        
        # 결정 일관성 확인
        decisions = [d.get("decision") for d in recent_decisions[-5:]]
        if len(set(decisions)) > 3:  # 결정이 너무 다양하면
            return True
        
        return False

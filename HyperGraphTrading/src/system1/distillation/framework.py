"""
지식 증류 프레임워크
System 2 (Teacher) → System 1 (Student)
"""
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import numpy as np


class KnowledgeDistillation:
    """지식 증류 클래스"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        """지식 증류 초기화"""
        self.temperature = temperature  # 증류 온도
        self.alpha = alpha  # 하드 레이블 vs 소프트 레이블 가중치
    
    def distill_policy(self, 
                      teacher_policy: Dict[str, Any],
                      student_model: nn.Module,
                      student_input: torch.Tensor) -> Dict[str, Any]:
        """정책 증류"""
        # Teacher 정책을 소프트 레이블로 변환
        teacher_output = self._policy_to_tensor(teacher_policy)
        
        # Student 모델 출력
        student_output = student_model(student_input)
        
        # 증류 손실 계산
        loss = self._compute_distillation_loss(
            teacher_output=teacher_output,
            student_output=student_output,
            temperature=self.temperature
        )
        
        return {
            "loss": loss.item(),
            "teacher_output": teacher_output,
            "student_output": student_output
        }
    
    def distill_knowledge(self,
                         teacher_knowledge: Dict[str, Any],
                         student_model: nn.Module,
                         student_input: torch.Tensor) -> Dict[str, Any]:
        """지식 증류 (특징 증류)"""
        # Teacher의 중간 특징 추출 (실제로는 Teacher 모델 필요)
        # 여기서는 간단한 구현
        
        # Student 특징
        if hasattr(student_model, 'get_features'):
            student_features = student_model.get_features(student_input)
        else:
            # 기본: 마지막 레이어 전 특징
            student_features = student_input
        
        # 특징 매칭 손실 (간단한 구현)
        loss = nn.MSELoss()(
            student_features,
            torch.zeros_like(student_features)  # 실제로는 Teacher 특징 사용
        )
        
        return {
            "loss": loss.item(),
            "student_features": student_features
        }
    
    def _policy_to_tensor(self, policy: Dict[str, Any]) -> torch.Tensor:
        """정책을 텐서로 변환"""
        decision = policy.get("decision", "HOLD")
        confidence = policy.get("confidence", 0.5)
        
        # 결정을 원-핫 벡터로 변환
        decision_map = {"BUY": 0, "SELL": 1, "HOLD": 2}
        decision_idx = decision_map.get(decision, 2)
        
        # 소프트 레이블 생성
        soft_label = torch.zeros(3)
        soft_label[decision_idx] = confidence
        # 나머지 확률 분산
        remaining = (1 - confidence) / 2
        for i in range(3):
            if i != decision_idx:
                soft_label[i] = remaining
        
        return soft_label
    
    def _compute_distillation_loss(self,
                                   teacher_output: torch.Tensor,
                                   student_output: torch.Tensor,
                                   temperature: float) -> torch.Tensor:
        """증류 손실 계산 (KL Divergence)"""
        # Temperature scaling
        teacher_soft = torch.softmax(teacher_output / temperature, dim=-1)
        student_soft = torch.softmax(student_output / temperature, dim=-1)
        
        # KL Divergence
        kl_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(student_soft + 1e-8),
            teacher_soft
        )
        
        return kl_loss * (temperature ** 2)
    
    def compute_total_loss(self,
                          distillation_loss: torch.Tensor,
                          task_loss: torch.Tensor,
                          alpha: float) -> torch.Tensor:
        """전체 손실 계산"""
        return alpha * distillation_loss + (1 - alpha) * task_loss


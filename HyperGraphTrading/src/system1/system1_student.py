"""
System 1 (Student) 메인 클래스
경량 실시간 모델
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path

from .model.architecture import LightweightTradingModel, SimplifiedTradingModel
from .inference.pipeline import InferencePipeline
from .distillation.framework import KnowledgeDistillation


class System1Student:
    """System 1 (Student) 시스템"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_type: str = "simplified",
                 device: str = "cpu"):
        """System 1 초기화"""
        self.device = torch.device(device)
        self.model_type = model_type
        
        # 모델 생성
        if model_type == "lightweight":
            self.model = LightweightTradingModel(
                input_dim=10,
                hidden_dim=64,
                output_dim=3
            )
        else:  # simplified
            self.model = SimplifiedTradingModel(
                input_dim=10,
                hidden_dim=32,
                output_dim=3
            )
        
        self.model.to(self.device)
        
        # 모델 로드 (있는 경우)
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # 추론 파이프라인
        self.pipeline = InferencePipeline(
            model=self.model,
            device=device
        )
        
        # 지식 증류 (학습용)
        self.distillation = KnowledgeDistillation()
    
    def infer(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """실시간 추론"""
        return self.pipeline.process_tick(tick_data)
    
    def train_from_teacher(self,
                          teacher_policies: List[Dict[str, Any]],
                          training_data: torch.Tensor,
                          epochs: int = 10,
                          learning_rate: float = 0.001) -> Dict[str, Any]:
        """Teacher 정책으로부터 학습"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        training_losses = []
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i, policy in enumerate(teacher_policies):
                if i >= len(training_data):
                    break
                
                # 입력 데이터
                x = training_data[i:i+1].to(self.device)
                
                # Teacher 정책을 타겟으로 변환
                decision = policy.get("decision", "HOLD")
                decision_map = {"BUY": 0, "SELL": 1, "HOLD": 2}
                target = torch.tensor([decision_map.get(decision, 2)], device=self.device)
                
                # 순전파
                output = self.model(x)
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output
                
                # 손실 계산
                loss = criterion(logits, target)
                
                # 지식 증류 손실 추가
                distillation_loss = self.distillation.distill_policy(
                    teacher_policy=policy,
                    student_model=self.model,
                    student_input=x
                )["loss"]
                
                total_loss = loss + 0.3 * distillation_loss
                
                # 역전파
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(teacher_policies)
            training_losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        
        return {
            "training_losses": training_losses,
            "final_loss": training_losses[-1] if training_losses else 0.0
        }
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "model_config": self._get_model_config()
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
    
    def _get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        if isinstance(self.model, LightweightTradingModel):
            return {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "output_dim": self.model.output_dim
            }
        elif isinstance(self.model, SimplifiedTradingModel):
            return {
                "input_dim": 10,
                "hidden_dim": 32,
                "output_dim": 3
            }
        return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계"""
        return self.pipeline.get_performance_stats()


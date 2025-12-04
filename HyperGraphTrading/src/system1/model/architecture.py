"""
System 1 경량 모델 아키텍처
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LightweightTradingModel(nn.Module):
    """경량 트레이딩 모델"""
    
    def __init__(self,
                 input_dim: int = 10,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 output_dim: int = 3,  # BUY, SELL, HOLD
                 dropout: float = 0.1):
        """모델 초기화"""
        super(LightweightTradingModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # LSTM 기반 시계열 인코더
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 어텐션 메커니즘 (선택적)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 의사결정 헤드
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 가치 함수 (선택적)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """순전파"""
        # LSTM 인코딩
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 마지막 타임스텝 사용
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # 어텐션 적용
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        attn_hidden = attn_out[:, -1, :]
        
        # 특징 결합
        combined = last_hidden + attn_hidden
        
        # 의사결정 출력
        decision_logits = self.decision_head(combined)
        
        # 가치 출력 (선택적)
        value = self.value_head(combined)
        
        return decision_logits, value
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """중간 특징 추출 (지식 증류용)"""
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return attn_out[:, -1, :]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """예측 (추론용)"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        return predictions


class SimplifiedTradingModel(nn.Module):
    """더 간단한 경량 모델 (최적화 버전)"""
    
    def __init__(self,
                 input_dim: int = 10,
                 hidden_dim: int = 32,
                 output_dim: int = 3):
        """간단한 모델 초기화"""
        super(SimplifiedTradingModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # 마지막 타임스텝만 사용
        if len(x.shape) == 3:
            x = x[:, -1, :]  # [batch, seq, features] -> [batch, features]
        
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """특징 추출"""
        if len(x.shape) == 3:
            x = x[:, -1, :]
        return self.encoder(x)


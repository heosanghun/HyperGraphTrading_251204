"""
리스크 관리 에이전트 (Risk Agent)
리스크 평가 및 관리
"""
from typing import Dict, List, Any
import numpy as np

from .base_agent import BaseAgent
from ...hypergraph import FinancialHypergraph, RelationType


class RiskAgent(BaseAgent):
    """리스크 관리 에이전트"""
    
    def __init__(self, hypergraph: FinancialHypergraph):
        """리스크 에이전트 초기화"""
        super().__init__(
            name="RiskManager",
            role="risk_management",
            hypergraph=hypergraph
        )
        self.risk_threshold = 0.7  # 리스크 임계값
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """리스크 분석"""
        symbol = context.get("symbol", "")
        
        risk_analysis = {
            "symbol": symbol,
            "risk_level": "LOW",
            "risk_score": 0.0,
            "factors": [],
            "recommendation": "PROCEED"
        }
        
        if not self.hypergraph:
            return risk_analysis
        
        # 하이퍼그래프에서 리스크 관련 엣지 찾기
        node = self.hypergraph.get_node(symbol)
        if not node:
            return risk_analysis
        
        # 리스크 하이퍼엣지 확인
        risk_edges = []
        for edge in self.hypergraph.edges.values():
            if edge.contains_node(symbol):
                # 리스크 관련 관계 확인
                if edge.relation_type in [RelationType.INFLUENCE, RelationType.MARKET_IMPACT]:
                    if edge.weight > self.risk_threshold:
                        risk_edges.append(edge)
        
        # 리스크 점수 계산
        if risk_edges:
            risk_score = np.mean([e.weight * e.confidence for e in risk_edges])
            risk_analysis["risk_score"] = float(risk_score)
            
            if risk_score > 0.8:
                risk_analysis["risk_level"] = "HIGH"
                risk_analysis["recommendation"] = "AVOID"
            elif risk_score > 0.5:
                risk_analysis["risk_level"] = "MEDIUM"
                risk_analysis["recommendation"] = "CAUTION"
            else:
                risk_analysis["risk_level"] = "LOW"
                risk_analysis["recommendation"] = "PROCEED"
            
            risk_analysis["factors"] = [
                {
                    "type": e.relation_type.value,
                    "weight": e.weight,
                    "confidence": e.confidence
                }
                for e in risk_edges
            ]
        
        self.update_memory(risk_analysis)
        return risk_analysis
    
    def generate_claim(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """리스크 기반 주장 생성"""
        claim = {
            "agent": self.name,
            "type": "risk_claim",
            "claim": "리스크 평가 완료",
            "evidence": evidence,
            "confidence": 0.8
        }
        
        # 리스크 근거 분석
        high_risk_count = sum(1 for e in evidence if e.get("risk_level") == "HIGH")
        
        if high_risk_count > 0:
            claim["claim"] = f"높은 리스크 감지 ({high_risk_count}개 요인)"
            claim["recommendation"] = "보수적 접근 권장"
            claim["confidence"] = 0.9
        else:
            claim["claim"] = "리스크 수준 양호"
            claim["recommendation"] = "HOLD"  # 리스크가 낮으면 관망
            claim["confidence"] = 0.7
        
        return claim


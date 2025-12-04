"""
System 2 (Teacher) 메인 클래스
하이퍼그래프 기반 근거 중심 토론 시스템
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

from .agents.analyst_agent import AnalystAgent
from .agents.risk_agent import RiskAgent
from .agents.strategy_agent import StrategyAgent
from .discussion.framework import DiscussionFramework
from .llm.interface import create_llm_interface
from .policy.extractor import PolicyExtractor
from ..hypergraph import FinancialHypergraph


class System2Teacher:
    """System 2 (Teacher) 시스템"""
    
    def __init__(self, 
                 hypergraph: FinancialHypergraph,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 use_llm: bool = False):
        """System 2 초기화"""
        self.hypergraph = hypergraph
        
        # 에이전트 생성
        self.analyst = AnalystAgent(hypergraph)
        self.risk_manager = RiskAgent(hypergraph)
        self.strategist = StrategyAgent(hypergraph)
        self.agents = [self.analyst, self.risk_manager, self.strategist]
        
        # 토론 프레임워크
        self.discussion_framework = DiscussionFramework(
            max_rounds=5,
            consensus_threshold=0.7
        )
        
        # LLM 인터페이스 (선택적)
        self.use_llm = use_llm
        if use_llm:
            try:
                self.llm = create_llm_interface(provider=llm_provider, model=llm_model)
            except Exception as e:
                print(f"LLM 초기화 실패 (선택적): {e}")
                self.llm = None
        else:
            self.llm = None
        
        # 정책 추출기
        self.policy_extractor = PolicyExtractor()
    
    def generate_policy(self, 
                       symbol: str,
                       date: str,
                       use_llm: bool = False) -> Dict[str, Any]:
        """정책 생성"""
        print(f">>> System 2: {symbol} 정책 생성 시작 ({date})")
        
        # 컨텍스트 준비
        context = {
            "symbol": symbol,
            "date": date,
            "hypergraph": self.hypergraph
        }
        
        # 각 에이전트 분석
        print("  - 에이전트 분석 중...")
        market_analysis = self.analyst.analyze(context)
        risk_analysis = self.risk_manager.analyze(context)
        
        context["market_analysis"] = market_analysis
        context["risk_analysis"] = risk_analysis
        
        strategy_analysis = self.strategist.analyze(context)
        context["strategy_analysis"] = strategy_analysis
        
        # 토론 시작
        print("  - 토론 진행 중...")
        topic = f"{symbol} 트레이딩 결정 ({date})"
        discussion = self.discussion_framework.initiate_discussion(
            topic=topic,
            agents=self.agents,
            hypergraph=self.hypergraph
        )
        
        # LLM 분석 (선택적)
        if use_llm and self.llm:
            print("  - LLM 분석 중...")
            hypergraph_data = {
                "total_nodes": len(self.hypergraph.nodes),
                "total_edges": len(self.hypergraph.edges),
                "type_correlations": {}
            }
            llm_analysis = self.llm.generate_analysis(hypergraph_data, context)
            discussion["llm_analysis"] = llm_analysis
        
        # 정책 추출
        print("  - 정책 추출 중...")
        policy = self.policy_extractor.extract_policy(discussion)
        
        # 정책 검증
        validation = self.policy_extractor.validate_policy(policy)
        policy["validation"] = validation
        
        if not validation["valid"]:
            print(f"  [WARNING] 정책 검증 실패: {validation['errors']}")
        
        print(f"<<< System 2: 정책 생성 완료 (결정: {policy['decision']}, 신뢰도: {policy['confidence']:.2f})")
        
        return {
            "policy": policy,
            "discussion": discussion,
            "analyses": {
                "market": market_analysis,
                "risk": risk_analysis,
                "strategy": strategy_analysis
            }
        }


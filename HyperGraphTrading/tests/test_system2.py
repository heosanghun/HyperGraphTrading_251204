"""
System 2 테스트
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hypergraph import FinancialHypergraph, HyperNode, HyperEdge, NodeType, RelationType
from src.system2 import System2Teacher, AnalystAgent, RiskAgent, StrategyAgent
from datetime import datetime


@pytest.fixture
def sample_hypergraph():
    """샘플 하이퍼그래프 생성"""
    hypergraph = FinancialHypergraph()
    
    # 노드 추가
    node1 = HyperNode(id="AAPL", type=NodeType.STOCK, features={"price": 150.0})
    node2 = HyperNode(id="MSFT", type=NodeType.STOCK, features={"price": 300.0})
    
    hypergraph.add_node(node1)
    hypergraph.add_node(node2)
    
    # 엣지 추가
    edge = HyperEdge(
        nodes=[node1, node2],
        weight=0.7,
        relation_type=RelationType.CORRELATION
    )
    hypergraph.add_hyperedge(edge)
    
    return hypergraph


def test_analyst_agent(sample_hypergraph):
    """분석 에이전트 테스트"""
    agent = AnalystAgent(sample_hypergraph)
    
    context = {"symbol": "AAPL", "date": "2023-06-01"}
    analysis = agent.analyze(context)
    
    assert "symbol" in analysis
    assert analysis["symbol"] == "AAPL"
    assert "findings" in analysis


def test_risk_agent(sample_hypergraph):
    """리스크 에이전트 테스트"""
    agent = RiskAgent(sample_hypergraph)
    
    context = {"symbol": "AAPL"}
    risk_analysis = agent.analyze(context)
    
    assert "risk_level" in risk_analysis
    assert risk_analysis["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_strategy_agent(sample_hypergraph):
    """전략 에이전트 테스트"""
    agent = StrategyAgent(sample_hypergraph)
    
    context = {
        "symbol": "AAPL",
        "market_analysis": {"findings": []},
        "risk_analysis": {"risk_level": "LOW"}
    }
    strategy = agent.analyze(context)
    
    assert "action" in strategy
    assert strategy["action"] in ["BUY", "SELL", "HOLD"]


def test_system2_teacher(sample_hypergraph):
    """System 2 Teacher 테스트"""
    teacher = System2Teacher(sample_hypergraph, use_llm=False)
    
    policy_result = teacher.generate_policy("AAPL", "2023-06-01", use_llm=False)
    
    assert "policy" in policy_result
    assert "decision" in policy_result["policy"]
    assert policy_result["policy"]["decision"] in ["BUY", "SELL", "HOLD"]


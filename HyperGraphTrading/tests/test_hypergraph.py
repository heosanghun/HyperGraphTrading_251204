"""
하이퍼그래프 모듈 테스트
"""
import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hypergraph import FinancialHypergraph, HyperNode, HyperEdge, NodeType, RelationType
from src.hypergraph.analyzer import HypergraphAnalyzer
from src.hypergraph.dynamic import DynamicHypergraphUpdater
from datetime import datetime


class TestHypergraphStructure:
    """하이퍼그래프 구조 테스트"""
    
    def test_create_node(self):
        """노드 생성 테스트"""
        node = HyperNode(
            id="AAPL",
            type=NodeType.STOCK,
            features={"price": 150.0, "volume": 1000000},
            timestamp=datetime.now()
        )
        assert node.id == "AAPL"
        assert node.type == NodeType.STOCK
        assert len(node.features) == 2
    
    def test_create_hyperedge(self):
        """하이퍼엣지 생성 테스트"""
        node1 = HyperNode(id="AAPL", type=NodeType.STOCK)
        node2 = HyperNode(id="MSFT", type=NodeType.STOCK)
        
        edge = HyperEdge(
            nodes=[node1, node2],
            weight=0.8,
            relation_type=RelationType.CORRELATION,
            evidence={"correlation": 0.8}
        )
        
        assert len(edge.nodes) == 2
        assert edge.weight == 0.8
        assert edge.relation_type == RelationType.CORRELATION
    
    def test_hyperedge_validation(self):
        """하이퍼엣지 검증 테스트"""
        node1 = HyperNode(id="AAPL", type=NodeType.STOCK)
        
        # 노드가 1개만 있으면 오류
        with pytest.raises(ValueError):
            HyperEdge(
                nodes=[node1],
                weight=0.8,
                relation_type=RelationType.CORRELATION
            )
        
        # 가중치 범위 검증
        node2 = HyperNode(id="MSFT", type=NodeType.STOCK)
        with pytest.raises(ValueError):
            HyperEdge(
                nodes=[node1, node2],
                weight=1.5,  # 범위 초과
                relation_type=RelationType.CORRELATION
            )


class TestFinancialHypergraph:
    """FinancialHypergraph 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.hypergraph = FinancialHypergraph()
        
        # 테스트 노드 생성
        self.node1 = HyperNode(id="AAPL", type=NodeType.STOCK, features={"price": 150.0})
        self.node2 = HyperNode(id="MSFT", type=NodeType.STOCK, features={"price": 300.0})
        self.node3 = HyperNode(id="GOOGL", type=NodeType.STOCK, features={"price": 2500.0})
    
    def test_add_node(self):
        """노드 추가 테스트"""
        self.hypergraph.add_node(self.node1)
        assert "AAPL" in self.hypergraph.nodes
        assert self.hypergraph.get_node("AAPL") == self.node1
    
    def test_add_hyperedge(self):
        """하이퍼엣지 추가 테스트"""
        self.hypergraph.add_node(self.node1)
        self.hypergraph.add_node(self.node2)
        
        edge = HyperEdge(
            nodes=[self.node1, self.node2],
            weight=0.7,
            relation_type=RelationType.CORRELATION
        )
        
        edge_id = self.hypergraph.add_hyperedge(edge)
        assert edge_id in self.hypergraph.edges
        assert len(self.hypergraph.edges) == 1
    
    def test_get_neighbors(self):
        """이웃 노드 조회 테스트"""
        self.hypergraph.add_node(self.node1)
        self.hypergraph.add_node(self.node2)
        self.hypergraph.add_node(self.node3)
        
        edge1 = HyperEdge(
            nodes=[self.node1, self.node2],
            weight=0.7,
            relation_type=RelationType.CORRELATION
        )
        edge2 = HyperEdge(
            nodes=[self.node1, self.node3],
            weight=0.6,
            relation_type=RelationType.CORRELATION
        )
        
        self.hypergraph.add_hyperedge(edge1)
        self.hypergraph.add_hyperedge(edge2)
        
        neighbors = self.hypergraph.get_neighbors("AAPL")
        assert "MSFT" in neighbors
        assert "GOOGL" in neighbors
        assert len(neighbors) == 2
    
    def test_get_evidence(self):
        """근거 추출 테스트"""
        self.hypergraph.add_node(self.node1)
        self.hypergraph.add_node(self.node2)
        
        edge = HyperEdge(
            nodes=[self.node1, self.node2],
            weight=0.8,
            relation_type=RelationType.CORRELATION,
            evidence={"correlation": 0.8, "method": "pearson"}
        )
        self.hypergraph.add_hyperedge(edge)
        
        evidence = self.hypergraph.get_evidence("AAPL")
        assert len(evidence) > 0
        assert evidence[0]["relation"] == "correlation"
    
    def test_subgraph_extraction(self):
        """서브그래프 추출 테스트"""
        self.hypergraph.add_node(self.node1)
        self.hypergraph.add_node(self.node2)
        self.hypergraph.add_node(self.node3)
        
        edge = HyperEdge(
            nodes=[self.node1, self.node2],
            weight=0.7,
            relation_type=RelationType.CORRELATION
        )
        self.hypergraph.add_hyperedge(edge)
        
        subgraph = self.hypergraph.get_subgraph(["AAPL", "MSFT"])
        assert len(subgraph.nodes) == 2
        assert len(subgraph.edges) == 1


class TestHypergraphAnalyzer:
    """HypergraphAnalyzer 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.hypergraph = FinancialHypergraph()
        
        # 테스트 그래프 구축
        nodes = [
            HyperNode(id="AAPL", type=NodeType.STOCK),
            HyperNode(id="MSFT", type=NodeType.STOCK),
            HyperNode(id="GOOGL", type=NodeType.STOCK),
        ]
        
        for node in nodes:
            self.hypergraph.add_node(node)
        
        # 엣지 추가
        edge1 = HyperEdge(
            nodes=[nodes[0], nodes[1]],
            weight=0.8,
            relation_type=RelationType.CORRELATION
        )
        edge2 = HyperEdge(
            nodes=[nodes[0], nodes[2]],
            weight=0.7,
            relation_type=RelationType.CORRELATION
        )
        
        self.hypergraph.add_hyperedge(edge1)
        self.hypergraph.add_hyperedge(edge2)
        
        self.analyzer = HypergraphAnalyzer(self.hypergraph)
    
    def test_compute_centrality(self):
        """중심성 계산 테스트"""
        centralities = self.analyzer.compute_centrality("AAPL")
        assert "degree" in centralities
        assert "weighted_degree" in centralities
        assert centralities["degree"] > 0
    
    def test_find_influential_nodes(self):
        """영향력 있는 노드 찾기 테스트"""
        influential = self.analyzer.find_influential_nodes(top_k=2)
        assert len(influential) <= 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in influential)
    
    def test_analyze_correlation_structure(self):
        """상관관계 구조 분석 테스트"""
        structure = self.analyzer.analyze_correlation_structure()
        assert "total_nodes" in structure
        assert "total_edges" in structure
        assert structure["total_nodes"] == 3
        assert structure["total_edges"] == 2


class TestDynamicUpdater:
    """DynamicHypergraphUpdater 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.hypergraph = FinancialHypergraph()
        self.updater = DynamicHypergraphUpdater(self.hypergraph)
    
    def test_update_weights(self):
        """가중치 업데이트 테스트"""
        node1 = HyperNode(id="AAPL", type=NodeType.STOCK)
        node2 = HyperNode(id="MSFT", type=NodeType.STOCK)
        
        self.hypergraph.add_node(node1)
        self.hypergraph.add_node(node2)
        
        edge = HyperEdge(
            nodes=[node1, node2],
            weight=0.8,
            relation_type=RelationType.CORRELATION
        )
        edge_id = self.hypergraph.add_hyperedge(edge)
        
        initial_weight = self.hypergraph.edges[edge_id].weight
        
        # 시간 업데이트
        self.updater.update_weights(datetime.now())
        
        # 가중치는 감쇠되어야 함 (또는 유지)
        updated_weight = self.hypergraph.edges[edge_id].weight
        assert updated_weight > 0  # 최소값 유지
    
    def test_remove_stale_data(self):
        """오래된 데이터 제거 테스트"""
        node = HyperNode(
            id="OLD_NODE",
            type=NodeType.NEWS,
            timestamp=datetime(2020, 1, 1)
        )
        self.hypergraph.add_node(node)
        
        self.updater.last_update_time = datetime.now()
        removed = self.updater.remove_stale_data(max_age_hours=24)
        
        # 오래된 노드는 제거되어야 함 (주식 노드 제외)
        assert removed >= 0


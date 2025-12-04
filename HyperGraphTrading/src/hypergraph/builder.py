"""
금융 하이퍼그래프 구축 모듈
"""
import networkx as nx
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import numpy as np
import pandas as pd

from .structure import HyperNode, HyperEdge, NodeType, RelationType
from .transfer_entropy import verify_hyperedge_causality


class FinancialHypergraph:
    """금융 하이퍼그래프 클래스"""
    
    def __init__(self):
        """하이퍼그래프 초기화"""
        # 노드 저장 (id -> HyperNode)
        self.nodes: Dict[str, HyperNode] = {}
        
        # 하이퍼엣지 저장 (edge_id -> HyperEdge)
        self.edges: Dict[str, HyperEdge] = {}
        
        # 노드 타입별 인덱스
        self.node_index: Dict[NodeType, Set[str]] = {
            node_type: set() for node_type in NodeType
        }
        
        # NetworkX 그래프 (하이퍼엣지 시뮬레이션용)
        self.nx_graph = nx.Graph()
        
    def add_node(self, node: HyperNode) -> None:
        """노드 추가"""
        self.nodes[node.id] = node
        self.node_index[node.type].add(node.id)
        self.nx_graph.add_node(node.id, **node.features, type=node.type.value)
    
    def get_node(self, node_id: str) -> Optional[HyperNode]:
        """노드 조회"""
        return self.nodes.get(node_id)
    
    def add_hyperedge(self, edge: HyperEdge, verify_causality: bool = False) -> str:
        """하이퍼엣지 추가"""
        # 전이 엔트로피 검증 (선택적)
        if verify_causality:
            node_ids = edge.get_node_ids()
            is_valid, te_score = verify_hyperedge_causality(self, node_ids, theta=2.0)
            if not is_valid:
                # 검증 실패 시 신뢰도 낮춤
                edge.confidence = min(edge.confidence, 0.3)
            else:
                # 검증 성공 시 신뢰도 향상
                edge.confidence = min(edge.confidence + 0.2, 1.0)
                edge.evidence["transfer_entropy"] = te_score
        
        edge_id = self._generate_edge_id(edge)
        self.edges[edge_id] = edge
        
        # NetworkX 그래프에 클리크로 추가 (하이퍼엣지 시뮬레이션)
        node_ids = edge.get_node_ids()
        if len(node_ids) >= 2:
            # 완전 그래프로 연결 (하이퍼엣지 표현)
            for i, node1 in enumerate(node_ids):
                for node2 in node_ids[i+1:]:
                    self.nx_graph.add_edge(
                        node1, node2,
                        weight=edge.weight,
                        relation_type=edge.relation_type.value,
                        edge_id=edge_id,
                        confidence=edge.confidence
                    )
        
        return edge_id
    
    def _generate_edge_id(self, edge: HyperEdge) -> str:
        """엣지 ID 생성"""
        node_ids = sorted(edge.get_node_ids())
        return f"{edge.relation_type.value}_{'_'.join(node_ids)}"
    
    def compute_correlation(self, node_ids: List[str], method: str = "pearson") -> float:
        """노드 간 상관관계 계산"""
        if len(node_ids) < 2:
            return 0.0
        
        # 노드의 특징 데이터 추출
        features_list = []
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                # 다양한 특징 데이터 소스 시도
                if 'price_data' in node.features:
                    data = node.features['price_data']
                elif 'close' in node.features:
                    data = node.features['close']
                elif isinstance(node.features, dict) and len(node.features) > 0:
                    # 첫 번째 숫자 리스트 찾기
                    for key, value in node.features.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], (int, float)):
                                data = value
                                break
                    else:
                        continue
                else:
                    continue
                
                if isinstance(data, list) and len(data) > 1:
                    features_list.append(data)
        
        if len(features_list) < 2:
            return 0.0
        
        # 길이 맞추기
        min_len = min(len(f) for f in features_list)
        features_list = [f[:min_len] for f in features_list]
        
        # 상관관계 계산
        try:
            if method == "pearson":
                # 피어슨 상관계수
                corr_matrix = np.corrcoef(features_list)
                # 평균 상관계수 반환 (대각선 제외)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                if np.any(mask):
                    return float(np.mean(corr_matrix[mask]))
                else:
                    return 0.0
            else:
                return 0.0
        except Exception as e:
            return 0.0
    
    def update_dynamic(self, timestamp: datetime) -> None:
        """동적 업데이트 (시점별 그래프 업데이트)"""
        # 타임스탬프 기반 노드/엣지 필터링
        # 오래된 데이터 제거 또는 가중치 조정
        pass
    
    def get_evidence(self, query: str) -> List[Dict[str, Any]]:
        """하이퍼그래프에서 근거 추출"""
        evidence_list = []
        
        # 쿼리와 관련된 노드 찾기
        query_lower = query.lower()
        relevant_nodes = []
        
        for node_id, node in self.nodes.items():
            if query_lower in node_id.lower() or \
               any(query_lower in str(v).lower() for v in node.features.values()):
                relevant_nodes.append(node)
        
        # 관련 노드를 포함하는 하이퍼엣지 찾기
        for edge_id, edge in self.edges.items():
            for node in relevant_nodes:
                if edge.contains_node(node.id):
                    evidence_list.append({
                        "edge_id": edge_id,
                        "nodes": edge.get_node_ids(),
                        "relation": edge.relation_type.value,
                        "weight": edge.weight,
                        "confidence": edge.confidence,
                        "evidence": edge.evidence
                    })
                    break
        
        return evidence_list
    
    def get_neighbors(self, node_id: str, relation_type: Optional[RelationType] = None) -> List[str]:
        """노드의 이웃 노드 조회"""
        neighbors = set()
        
        for edge in self.edges.values():
            if edge.contains_node(node_id):
                for node in edge.nodes:
                    if node.id != node_id:
                        if relation_type is None or edge.relation_type == relation_type:
                            neighbors.add(node.id)
        
        return list(neighbors)
    
    def get_subgraph(self, node_ids: List[str]) -> 'FinancialHypergraph':
        """서브그래프 추출"""
        subgraph = FinancialHypergraph()
        
        # 노드 추가
        for node_id in node_ids:
            node = self.get_node(node_id)
            if node:
                subgraph.add_node(node)
        
        # 관련 엣지 추가
        for edge in self.edges.values():
            edge_node_ids = edge.get_node_ids()
            if all(nid in node_ids for nid in edge_node_ids):
                subgraph.add_hyperedge(edge)
        
        return subgraph
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (직렬화)"""
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "type": node.type.value,
                    "features": node.features,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": {
                edge_id: {
                    "node_ids": edge.get_node_ids(),
                    "weight": edge.weight,
                    "relation_type": edge.relation_type.value,
                    "evidence": edge.evidence,
                    "confidence": edge.confidence
                }
                for edge_id, edge in self.edges.items()
            }
        }


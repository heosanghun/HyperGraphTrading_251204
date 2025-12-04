"""
하이퍼그래프 구축 예제 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hypergraph import FinancialHypergraph, HyperNode, HyperEdge, NodeType, RelationType
from src.data.collector import DataCollector
from datetime import datetime
import pandas as pd


def build_example_hypergraph():
    """예제 하이퍼그래프 구축"""
    print("="*60)
    print("하이퍼그래프 구축 예제")
    print("="*60)
    
    # 하이퍼그래프 생성
    hypergraph = FinancialHypergraph()
    
    # 1. 주가 데이터 수집
    print("\n1. 주가 데이터 수집 중...")
    collector = DataCollector()
    symbols = ["AAPL", "MSFT", "GOOGL"]
    price_data = collector.collect_price_data(
        symbols=symbols,
        start_date="2023-01-01",
        end_date="2023-12-31",
        interval="1d"
    )
    
    # 2. 노드 생성
    print("\n2. 노드 생성 중...")
    for symbol, df in price_data.items():
        # 주식 노드
        node = HyperNode(
            id=symbol,
            type=NodeType.STOCK,
            features={
                "price_data": df['close'].tolist()[-30:],  # 최근 30일
                "volume": df['volume'].tolist()[-30:],
                "symbol": symbol
            },
            timestamp=datetime.now()
        )
        hypergraph.add_node(node)
        print(f"  ✅ 노드 추가: {symbol}")
    
    # 3. 하이퍼엣지 생성 (상관관계 기반)
    print("\n3. 하이퍼엣지 생성 중...")
    symbols_list = list(price_data.keys())
    
    for i, symbol1 in enumerate(symbols_list):
        for symbol2 in symbols_list[i+1:]:
            # 상관관계 계산
            data1 = price_data[symbol1]['close'].tolist()[-30:]
            data2 = price_data[symbol2]['close'].tolist()[-30:]
            
            if len(data1) == len(data2):
                correlation = pd.Series(data1).corr(pd.Series(data2))
                
                if abs(correlation) > 0.3:  # 임계값
                    node1 = hypergraph.get_node(symbol1)
                    node2 = hypergraph.get_node(symbol2)
                    
                    edge = HyperEdge(
                        nodes=[node1, node2],
                        weight=abs(correlation),
                        relation_type=RelationType.CORRELATION,
                        evidence={
                            "correlation": correlation,
                            "method": "pearson"
                        },
                        confidence=0.8 if abs(correlation) > 0.7 else 0.5
                    )
                    
                    edge_id = hypergraph.add_hyperedge(edge)
                    print(f"  ✅ 엣지 추가: {symbol1} <-> {symbol2} (상관계수: {correlation:.3f})")
    
    # 4. 그래프 정보 출력
    print("\n4. 하이퍼그래프 정보:")
    print(f"  노드 수: {len(hypergraph.nodes)}")
    print(f"  엣지 수: {len(hypergraph.edges)}")
    
    # 5. 분석
    print("\n5. 하이퍼그래프 분석:")
    from src.hypergraph.analyzer import HypergraphAnalyzer
    
    analyzer = HypergraphAnalyzer(hypergraph)
    
    # 중심성 분석
    for symbol in symbols:
        centralities = analyzer.compute_centrality(symbol)
        print(f"\n  {symbol} 중심성:")
        for metric, value in centralities.items():
            print(f"    {metric}: {value:.4f}")
    
    # 영향력 있는 노드
    influential = analyzer.find_influential_nodes(top_k=3)
    print(f"\n  영향력 있는 노드 (상위 3개):")
    for node_id, score in influential:
        print(f"    {node_id}: {score:.4f}")
    
    # 상관관계 구조
    structure = analyzer.analyze_correlation_structure()
    print(f"\n  상관관계 구조:")
    print(f"    노드 분포: {structure['type_distribution']}")
    print(f"    총 노드: {structure['total_nodes']}")
    print(f"    총 엣지: {structure['total_edges']}")
    
    print("\n" + "="*60)
    print("하이퍼그래프 구축 완료!")
    print("="*60)
    
    return hypergraph


if __name__ == "__main__":
    hypergraph = build_example_hypergraph()


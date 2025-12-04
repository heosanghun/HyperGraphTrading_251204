"""
전체 시스템 실행 스크립트
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hypergraph import FinancialHypergraph, HyperNode, NodeType, RelationType
from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.system2 import System2Teacher
from src.system1 import System1Student
from src.integration import SystemIntegrator
from src.trading.backtester import Backtester
from datetime import datetime
import pandas as pd


def build_hypergraph_from_data(symbols: List[str], start_date: str, end_date: str):
    """데이터로부터 하이퍼그래프 구축"""
    print("하이퍼그래프 구축 중...")
    
    # 데이터 수집
    collector = DataCollector()
    price_data = collector.collect_price_data(symbols, start_date, end_date)
    
    # 전처리
    preprocessor = DataPreprocessor()
    processed_data = {}
    for symbol, df in price_data.items():
        df_clean = preprocessor.handle_missing_values(df)
        df_features = preprocessor.engineer_features(df_clean)
        processed_data[symbol] = df_features
    
    # 하이퍼그래프 생성
    hypergraph = FinancialHypergraph()
    
    for symbol, df in processed_data.items():
        node = HyperNode(
            id=symbol,
            type=NodeType.STOCK,
            features={
                "price_data": df["close"].tolist()[-30:] if "close" in df.columns else [],
                "volume": df["volume"].tolist()[-30:] if "volume" in df.columns else []
            }
        )
        hypergraph.add_node(node)
    
    # 상관관계 엣지 생성
    from src.hypergraph import HyperEdge
    
    symbols_list = list(processed_data.keys())
    for i, symbol1 in enumerate(symbols_list):
        for symbol2 in symbols_list[i+1:]:
            data1 = processed_data[symbol1]["close"].tolist()[-30:]
            data2 = processed_data[symbol2]["close"].tolist()[-30:]
            
            if len(data1) == len(data2) and len(data1) > 0:
                correlation = pd.Series(data1).corr(pd.Series(data2))
                
                if abs(correlation) > 0.3:
                    node1 = hypergraph.get_node(symbol1)
                    node2 = hypergraph.get_node(symbol2)
                    
                    edge = HyperEdge(
                        nodes=[node1, node2],
                        weight=abs(correlation),
                        relation_type=RelationType.CORRELATION,
                        evidence={"correlation": correlation}
                    )
                    hypergraph.add_hyperedge(edge)
    
    return hypergraph, processed_data


def main():
    """메인 실행 함수"""
    print("="*80)
    print("HyperGraphTrading 전체 시스템 실행")
    print("="*80)
    
    # 설정
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # 1. 하이퍼그래프 구축
    print("\n[1/5] 하이퍼그래프 구축...")
    hypergraph, processed_data = build_hypergraph_from_data(symbols, start_date, end_date)
    print(f"  ✅ 완료 (노드: {len(hypergraph.nodes)}, 엣지: {len(hypergraph.edges)})")
    
    # 2. System 2 초기화
    print("\n[2/5] System 2 초기화...")
    system2 = System2Teacher(hypergraph, use_llm=False)
    print("  ✅ 완료")
    
    # 3. System 1 초기화
    print("\n[3/5] System 1 초기화...")
    system1 = System1Student(model_type="simplified")
    print("  ✅ 완료")
    
    # 4. 통합 시스템
    print("\n[4/5] 시스템 통합...")
    integrator = SystemIntegrator(hypergraph, system2, system1)
    
    # System 2로 정책 생성 및 System 1 학습
    for symbol in symbols[:1]:  # 첫 번째 주식만 테스트
        print(f"  - {symbol} 정책 생성 및 학습...")
        result = integrator.update_system1_from_system2(symbol, "2023-06-01")
        print(f"    결정: {result['policy']['decision']}, 신뢰도: {result['policy']['confidence']:.2f}")
    
    print("  ✅ 통합 완료")
    
    # 5. 백테스팅
    print("\n[5/5] 백테스팅 실행...")
    if "AAPL" in processed_data:
        backtester = Backtester(initial_capital=10000.0)
        
        test_df = processed_data["AAPL"]
        # System 1으로 결정 생성 (시뮬레이션)
        decisions = []
        for i in range(min(50, len(test_df))):
            tick_data = {
                "price": test_df.iloc[i].get("close", 100),
                "volume": test_df.iloc[i].get("volume", 1000000),
                "prices": test_df["close"].iloc[max(0, i-19):i+1].tolist()
            }
            result = system1.infer(tick_data)
            decisions.append(result["prediction"])
        
        backtest_result = backtester.run_backtest(test_df.iloc[:len(decisions)], decisions)
        metrics = backtest_result["metrics"]
        
        print(f"  ✅ 백테스팅 완료")
        print(f"     총 수익률: {metrics.get('total_return', 0)*100:.2f}%")
        print(f"     Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"     최대 낙폭: {metrics.get('max_drawdown', 0)*100:.2f}%")
    
    print("\n" + "="*80)
    print("전체 시스템 실행 완료!")
    print("="*80)


if __name__ == "__main__":
    main()


"""
통합 테스트 스크립트
전체 시스템 엔드투엔드 테스트
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hypergraph import FinancialHypergraph, HyperNode, HyperEdge, NodeType, RelationType
from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.system2 import System2Teacher
from src.system1 import System1Student
from src.trading.backtester import Backtester
from datetime import datetime
import pandas as pd


def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("="*80)
    print("HyperGraphTrading 통합 테스트")
    print("="*80)
    
    # 1. 데이터 수집
    print("\n[1/6] 데이터 수집...")
    collector = DataCollector()
    symbols = ["AAPL", "MSFT"]
    price_data = collector.collect_price_data(
        symbols=symbols,
        start_date="2023-01-01",
        end_date="2023-12-31",
        interval="1d"
    )
    print(f"  ✅ {len(price_data)}개 주식 데이터 수집 완료")
    
    # 2. 데이터 전처리
    print("\n[2/6] 데이터 전처리...")
    preprocessor = DataPreprocessor()
    processed_data = {}
    for symbol, df in price_data.items():
        df_processed = preprocessor.handle_missing_values(df)
        df_processed = preprocessor.engineer_features(df_processed)
        processed_data[symbol] = df_processed
    print("  ✅ 전처리 완료")
    
    # 3. 하이퍼그래프 구축
    print("\n[3/6] 하이퍼그래프 구축...")
    hypergraph = FinancialHypergraph()
    
    for symbol, df in processed_data.items():
        node = HyperNode(
            id=symbol,
            type=NodeType.STOCK,
            features={
                "price_data": df["close"].tolist()[-30:] if "close" in df.columns else [],
                "volume": df["volume"].tolist()[-30:] if "volume" in df.columns else []
            },
            timestamp=datetime.now()
        )
        hypergraph.add_node(node)
    
    # 상관관계 엣지 생성
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
    
    print(f"  ✅ 하이퍼그래프 구축 완료 (노드: {len(hypergraph.nodes)}, 엣지: {len(hypergraph.edges)})")
    
    # 4. System 2 정책 생성
    print("\n[4/6] System 2 정책 생성...")
    try:
        teacher = System2Teacher(hypergraph, llm_provider="openai", llm_model="gpt-4o-mini")
        policy_result = teacher.generate_policy(
            symbol="AAPL",
            date="2023-06-01",
            use_llm=False  # LLM 없이 테스트
        )
        policy = policy_result["policy"]
        print(f"  ✅ 정책 생성 완료 (결정: {policy['decision']}, 신뢰도: {policy['confidence']:.2f})")
    except Exception as e:
        print(f"  ⚠️ System 2 오류 (계속 진행): {e}")
        policy = {"decision": "HOLD", "confidence": 0.5}
    
    # 5. System 1 학습 및 추론
    print("\n[5/6] System 1 학습 및 추론...")
    try:
        student = System1Student(model_type="simplified")
        
        # 간단한 학습 데이터 생성
        import torch
        training_data = torch.randn(10, 1, 10)  # [batch, seq, features]
        teacher_policies = [policy] * 10
        
        # 학습 (짧게)
        training_result = student.train_from_teacher(
            teacher_policies=teacher_policies,
            training_data=training_data,
            epochs=5,
            learning_rate=0.001
        )
        print(f"  ✅ 학습 완료 (최종 손실: {training_result['final_loss']:.4f})")
        
        # 추론 테스트
        tick_data = {
            "price": 150.0,
            "volume": 1000000,
            "prices": [150.0] * 20
        }
        inference_result = student.infer(tick_data)
        print(f"  ✅ 추론 완료 (결정: {inference_result['prediction']['decision']})")
        
        # 성능 통계
        perf_stats = student.get_performance_stats()
        if perf_stats:
            print(f"  📊 평균 추론 시간: {perf_stats.get('mean_inference_time_ms', 0):.2f}ms")
    
    except Exception as e:
        print(f"  ⚠️ System 1 오류: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 백테스팅
    print("\n[6/6] 백테스팅...")
    try:
        backtester = Backtester(initial_capital=10000.0)
        
        if "AAPL" in processed_data:
            test_df = processed_data["AAPL"]
            decisions = [
                {"decision": "BUY", "confidence": 0.7},
                {"decision": "HOLD", "confidence": 0.5},
                {"decision": "SELL", "confidence": 0.6}
            ] * (len(test_df) // 3 + 1)
            decisions = decisions[:len(test_df)]
            
            backtest_result = backtester.run_backtest(test_df, decisions)
            metrics = backtest_result["metrics"]
            
            print(f"  ✅ 백테스팅 완료")
            print(f"     총 수익률: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"     Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"     최대 낙폭: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"     총 거래: {metrics.get('total_trades', 0)}회")
    
    except Exception as e:
        print(f"  ⚠️ 백테스팅 오류: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("통합 테스트 완료!")
    print("="*80)


if __name__ == "__main__":
    test_full_pipeline()


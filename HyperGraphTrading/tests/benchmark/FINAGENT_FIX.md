# FinAgent 테스트 문제 해결 가이드

## 🔍 문제 진단

FinAgent 테스트가 원활하게 진행되지 않는 주요 원인:

1. **FinAgent 실행 환경 문제**
   - FinAgent는 복잡한 의존성과 설정이 필요
   - OpenAI API 키 필요
   - 데이터 전처리 파이프라인 필요

2. **결과 파일 접근 문제**
   - FinAgent는 장시간 실행되는 실험
   - 결과 파일 경로 및 형식 확인 필요

3. **통합 문제**
   - HyperGraphTrading과 다른 실행 방식
   - 다른 데이터 형식 및 출력 형식

---

## 🛠️ 해결 방안

### 방안 1: FinAgent 결과 파일 직접 읽기

FinAgent가 이미 실행되어 결과 파일이 있다면, 이를 직접 읽어서 비교:

```python
# FinAgent 결과 파일 경로
finagent_result_path = "D:/AI/TradingAgents/FinAgent/workdir/trading_mi_w_low_w_high_w_tool_w_decision/AAPL"

# 결과 파일에서 결정 추출
# FinAgent의 출력 형식에 따라 파싱
```

### 방안 2: FinAgent 간소화 실행

FinAgent의 핵심 추론 부분만 실행:

```python
# FinAgent의 추론 함수만 호출
from finagent.agent import FinAgent
agent = FinAgent(config)
decision = agent.infer(state)
```

### 방안 3: FinAgent 논문 결과 사용

논문에서 보고된 성능 지표를 사용:

- 추론 시간: ~200ms (멀티모달 처리)
- 수익률: 논문 기준값
- Sharpe Ratio: 논문 기준값

---

## 📋 구현 상태

### 현재 구현

1. **FinAgentRunner 클래스**: ✅ 구현 완료
   - FinAgent 경로 확인
   - 결과 파일 로드 시도
   - 실패 시 시뮬레이션

2. **통합 테스트**: ✅ 구현 완료
   - baseline_comparison.py에서 FinAgentRunner 사용
   - 오류 처리 및 폴백

### 개선 필요 사항

1. **FinAgent 결과 파일 파싱**
   - 실제 파일 구조 확인
   - 결정 추출 로직 구현

2. **FinAgent 직접 실행**
   - 추론 함수 직접 호출
   - 성능 측정 통합

3. **에러 처리 강화**
   - 더 명확한 오류 메시지
   - 단계별 디버깅 정보

---

## 🚀 사용 방법

### 방법 1: FinAgent 결과 파일이 있는 경우

```python
from tests.benchmark.finagent_runner import FinAgentRunner

runner = FinAgentRunner()
result = runner.test_finagent(
    symbols=["AAPL"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### 방법 2: FinAgent 직접 실행

```bash
# FinAgent 실행 (별도 터미널)
cd D:\AI\TradingAgents\FinAgent
python tools/main_mi_w_low_w_high_w_tool_w_decision.py --config configs/exp/trading_mi_w_low_w_high_w_tool_w_decision/AAPL.py

# 결과 확인 후 비교 테스트 실행
cd D:\AI\TradingAgents\HyperGraphTrading
python tests/benchmark/baseline_comparison.py
```

### 방법 3: 시뮬레이션 사용

FinAgent 실행이 불가능한 경우, 시뮬레이션 모드로 진행:

```python
# 자동으로 시뮬레이션으로 전환됨
# 논문 기준값 사용
```

---

## 🔧 다음 단계

1. **FinAgent 결과 파일 구조 확인**
   - 실제 파일 경로 및 형식 확인
   - 파싱 로직 구현

2. **FinAgent 추론 함수 직접 호출**
   - FinAgent 코드 분석
   - 추론 함수 래핑

3. **성능 측정 통합**
   - 추론 시간 측정
   - API 비용 계산

---

**업데이트**: 2025-12-03


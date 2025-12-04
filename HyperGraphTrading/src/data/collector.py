"""
데이터 수집 모듈
"""
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import time


class DataCollector:
    """데이터 수집 클래스"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_price_data(self, 
                          symbols: List[str],
                          start_date: str,
                          end_date: str,
                          interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """주가 데이터 수집"""
        price_data = {}
        
        for symbol in symbols:
            try:
                print(f"수집 중: {symbol}")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if not df.empty:
                    # 컬럼명 정규화
                    df = df.reset_index()
                    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                    
                    # 저장
                    save_path = self.data_dir / "prices" / f"{symbol}.csv"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(save_path, index=False)
                    
                    price_data[symbol] = df
                    print(f"  [OK] {symbol}: {len(df)}일 데이터 수집 완료")
                else:
                    print(f"  ⚠️ {symbol}: 데이터 없음")
                
                time.sleep(0.5)  # API 호출 제한 방지
                
            except Exception as e:
                print(f"  [ERROR] {symbol}: 오류 - {e}")
        
        return price_data
    
    def collect_news_data(self, symbols: List[str], days: int = 30) -> Dict[str, List[Dict]]:
        """뉴스 데이터 수집 (yfinance 기반, 제한적)"""
        # yfinance는 뉴스 데이터를 직접 제공하지 않음
        # 향후 다른 API로 확장 가능
        news_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # yfinance의 news 속성은 제한적
                news_data[symbol] = []
            except Exception as e:
                print(f"뉴스 수집 오류 ({symbol}): {e}")
        
        return news_data


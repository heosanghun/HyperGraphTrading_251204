"""
데이터 전처리 모듈
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        """전처리기 초기화"""
        pass
    
    def normalize_timeseries(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """시계열 데이터 정규화"""
        df_normalized = df.copy()
        
        for col in columns:
            if col in df_normalized.columns:
                # Z-score 정규화
                mean = df_normalized[col].mean()
                std = df_normalized[col].std()
                if std > 0:
                    df_normalized[col] = (df_normalized[col] - mean) / std
        
        return df_normalized
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """결측치 처리"""
        df_cleaned = df.copy()
        
        if method == "forward_fill":
            df_cleaned = df_cleaned.ffill()
        elif method == "backward_fill":
            df_cleaned = df_cleaned.bfill()
        elif method == "interpolate":
            df_cleaned = df_cleaned.interpolate()
        else:
            df_cleaned = df_cleaned.fillna(0)
        
        return df_cleaned
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특징 엔지니어링"""
        df_features = df.copy()
        
        # 가격 데이터가 있는 경우
        if "close" in df_features.columns or "Close" in df_features.columns:
            close_col = "close" if "close" in df_features.columns else "Close"
            
            # 이동평균
            df_features["ma_5"] = df_features[close_col].rolling(window=5).mean()
            df_features["ma_20"] = df_features[close_col].rolling(window=20).mean()
            
            # 변동성
            df_features["volatility"] = df_features[close_col].rolling(window=20).std()
            
            # 수익률
            df_features["returns"] = df_features[close_col].pct_change()
            
            # RSI (간단한 구현)
            delta = df_features[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            df_features["rsi"] = 100 - (100 / (1 + rs))
        
        return df_features
    
    def prepare_for_hypergraph(self, 
                               price_data: Dict[str, pd.DataFrame],
                               window_size: int = 30) -> Dict[str, np.ndarray]:
        """하이퍼그래프 구축을 위한 데이터 준비"""
        prepared_data = {}
        
        for symbol, df in price_data.items():
            # 최근 window_size일 데이터 추출
            if "close" in df.columns:
                prices = df["close"].tail(window_size).values
            elif "Close" in df.columns:
                prices = df["Close"].tail(window_size).values
            else:
                continue
            
            prepared_data[symbol] = prices
        
        return prepared_data


import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors


def march_split(df, val_size=0.2, random_state=42):
    # 월별로 데이터 분류
    df['month'] = df['datetime'].dt.month
    df['period'] = pd.cut(df['month'], bins=[0, 2, 3, 12], labels=['before_march', 'march', 'after_march'])

    # period (3월전 3월 3월이후 세 그룹으로 카테고리된 컬럼) 기준으로 계층 스플릿
    train, val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['period'])

    return train, val


def preprocess(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    strategy='median'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    주어진 훈련, 검증 데이터셋에 대해
    결측치 처리, 스케일링을 수행합니다.

    Parameters:
    x_train (DataFrame): 훈련 데이터셋
    x_valid (DataFrame): 검증 데이터셋
    strategy (str): 결측치 처리 방법

    Returns:
    Tuple[DataFrame, DataFrame]: 전처리된 훈련, 검증 데이터셋


    사용 예시
    x_train_processed, x_valid_processed = preprocess(
    x_train,
    x_valid,
    strategy
    )
    """
    tmp_x_train = x_train.copy()
    tmp_x_valid = x_valid.copy()

    tmp_x_train.reset_index(drop=True, inplace=True)
    tmp_x_valid.reset_index(drop=True, inplace=True)

    # 결측치 처리
    if strategy in ['bfill', 'ffill']:
        opposite = 'ffill' if strategy == 'bfill' else 'bfill'
        tmp_x_train = tmp_x_train.fillna(method=strategy).fillna(method=opposite)
        tmp_x_valid = tmp_x_valid.fillna(method=strategy).fillna(method=opposite)
    else:
        imputer = SimpleImputer(strategy=strategy)
        tmp_x_train = pd.DataFrame(imputer.fit_transform(tmp_x_train), columns=tmp_x_train.columns)
        tmp_x_valid = pd.DataFrame(imputer.transform(tmp_x_valid), columns=tmp_x_valid.columns)

    # 스케일링 평균 0 분산 1인 정규화
    scaler = StandardScaler()
    tmp_x_train = pd.DataFrame(scaler.fit_transform(tmp_x_train), columns=tmp_x_train.columns)
    tmp_x_valid = pd.DataFrame(scaler.transform(tmp_x_valid), columns=tmp_x_valid.columns)

    return tmp_x_train, tmp_x_valid


def preprocess_full(
    x_train: pd.DataFrame,
    strategy='median'
) -> pd.DataFrame:
    """
    주어진 훈련 데이터셋에 대해
    결측치 처리, 스케일링을 수행합니다.

    Parameters:
    x_train (DataFrame): 훈련 데이터셋
    strategy (str): 결측치 처리 방법

    Returns:
    DataFrame: 전처리된 훈련, 검증 데이터셋


    사용 예시
    x_train_processed = preprocess(
    x_train,
    strategy
    )
    """
    tmp_x_train = x_train.copy()

    tmp_x_train.reset_index(drop=True, inplace=True)

    # 결측치 처리
    if strategy in ['bfill', 'ffill']:
        opposite = 'ffill' if strategy == 'bfill' else 'bfill'
        tmp_x_train = tmp_x_train.fillna(method=strategy).fillna(method=opposite)
    else:
        imputer = SimpleImputer(strategy=strategy)
        tmp_x_train = pd.DataFrame(imputer.fit_transform(tmp_x_train), columns=tmp_x_train.columns)

    # 스케일링 평균 0 분산 1인 정규화
    scaler = StandardScaler()
    tmp_x_train = pd.DataFrame(scaler.fit_transform(tmp_x_train), columns=tmp_x_train.columns)

    return tmp_x_train


def quantile_transform(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    quantile_columns: List[str],
    n_quantiles: int,
    output_distribution: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    주어진 훈련, 검증 데이터셋에 대해
    Quantile Transformation을 수행합니다.

    Parameters:
    x_train (pd.DataFrame): 훈련 데이터셋
    x_valid (pd.DataFrame): 검증 데이터셋
    quantile_columns (List[str]): Quantile Transformation을 적용할 컬럼 리스트
    n_quantiles (int): QuantileTransformer의 n_quantiles 파라미터
    output_distribution (str): QuantileTransformer의 output_distribution 파라미터

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Quantile Transformation이 적용된 훈련, 검증 데이터셋

    사용 예시:
    x_train_processed, x_valid_processed = quantile_transform(
        x_train,
        x_valid,
        quantile_columns=['column1', 'column2', 'column3'],
        n_quantiles=1000,
        output_distribution='normal'
    )
    """
    tmp_x_train = x_train.copy()
    tmp_x_valid = x_valid.copy()

    # Quantile Transformation
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)
    tmp_x_train[quantile_columns] = qt.fit_transform(tmp_x_train[quantile_columns])
    tmp_x_valid[quantile_columns] = qt.transform(tmp_x_valid[quantile_columns])

    return tmp_x_train, tmp_x_valid


class TSMOTE:
    def __init__(self, k_neighbors=5, window_size=24):
        self.k = k_neighbors
        self.window_size = window_size
        self.smote = SMOTE(k_neighbors=k_neighbors, random_state=42)

    def fit_resample(self, X, y):
        # 시간 정보 추출
        times = X.index
        
        # 특성 데이터 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 시간 정보를 특성에 추가
        X_with_time = np.column_stack([X_scaled, times.astype(int) / 10**9])
        
        # 슬라이딩 윈도우 적용
        X_windowed, y_windowed = self._create_windows(X_with_time, y)
        
        # SMOTE 적용
        X_resampled, y_resampled = self.smote.fit_resample(X_windowed, y_windowed)
        
        # 윈도우 형태에서 원래 시계열 형태로 변환
        X_final, y_final = self._reconstruct_timeseries(X_resampled, y_resampled)
        
        # 스케일링 복원 및 시간 정보 재구성
        X_final_descaled = scaler.inverse_transform(X_final[:, :-1])
        times_final = pd.to_datetime(X_final[:, -1] * 10**9)
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame(X_final_descaled, columns=X.columns)
        result_df['date'] = times_final
        result_df.set_index('date', inplace=True)
        
        return result_df, y_final

    def _create_windows(self, X, y):
        X_windowed, y_windowed = [], []
        for i in range(len(X) - self.window_size + 1):
            X_windowed.append(X[i:i+self.window_size].flatten())
            y_windowed.append(y[i+self.window_size-1])
        return np.array(X_windowed), np.array(y_windowed)

    def _reconstruct_timeseries(self, X_windowed, y_windowed):
        feature_count = X_windowed.shape[1] // self.window_size
        X_reconstructed = []
        y_reconstructed = []
        for i in range(len(X_windowed)):
            if i == 0:
                X_reconstructed.extend(X_windowed[i].reshape(self.window_size, feature_count))
                y_reconstructed.extend([y_windowed[i]] * self.window_size)
            else:
                X_reconstructed.append(X_windowed[i][-feature_count:])
                y_reconstructed.append(y_windowed[i])
        return np.array(X_reconstructed), np.array(y_reconstructed)

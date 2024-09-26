import pandas as pd
import numpy as np
from typing import List, Tuple
import pywt


# 게시글에 있는 베이스라인 코드의 피처 엔지니어링 부분과 동일 
# (eda 에서 파악한 차이와 차이의 음수, 양수 여부를 새로운 피쳐로 생성)
def make_baseline_extra(df): 

    df = df.assign(
    liquidation_diff=df["liquidations_all_exchange_all_symbol_long_liquidations"] - df["liquidations_all_exchange_all_symbol_short_liquidations"],
    liquidation_usd_diff=df["liquidations_all_exchange_all_symbol_long_liquidations_usd"] - df["liquidations_all_exchange_all_symbol_short_liquidations_usd"],
    volume_diff=df["taker-buy-sell-stats_all_exchange_taker_buy_volume"] - df["taker-buy-sell-stats_all_exchange_taker_sell_ratio"],
    liquidation_diffg=np.sign(df["liquidations_all_exchange_all_symbol_long_liquidations"] - df["liquidations_all_exchange_all_symbol_short_liquidations"]),
    liquidation_usd_diffg=np.sign(df["liquidations_all_exchange_all_symbol_long_liquidations_usd"] - df["liquidations_all_exchange_all_symbol_short_liquidations_usd"]),
    volume_diffg=np.sign(df["taker-buy-sell-stats_all_exchange_taker_buy_volume"] - df["taker-buy-sell-stats_all_exchange_taker_sell_ratio"]),
    buy_sell_volume_ratio=df["taker-buy-sell-stats_all_exchange_taker_buy_volume"] / (df["taker-buy-sell-stats_all_exchange_taker_sell_ratio"] + 1))

    return df


def make_shift(
    df: pd.DataFrame,
    conti_cols: List[str],
    intervals: List[int],
) -> List[pd.Series]:
    """
    연속형 변수의 shift feature 생성
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
        intervals (List[int]): shifted intervals
    Return:
        List[pd.Series]
    """
    df_shift_dict = [
        df[conti_col].shift(interval).rename(f"{conti_col}_{interval}")
        for conti_col in conti_cols
        for interval in intervals
    ]
    return df_shift_dict


# 지수 이동 평균
def make_EMA(df: pd.DataFrame, col: List[str], span=2) -> pd.DataFrame:
    """
    지수 이동 평균을 새로운 피처로 추가.
    Args:
        df (pd.DataFrame): 데이터를 포함한 데이터프레임.
        col (List[str]): 지수 이동 평균 피처를 추가하고자 하는 열 이름 목록.
        span (int): 지수 이동 평균의 기간. 기본값은 2.
    Returns:
        pd.DataFrame: 지수 이동 평균 피처가 추가된 컬럼들의 데이터프레임 반환.
    """
    return df[col].ewm(span=span).mean()


# Wavlet Transform
def make_WT(df: pd.DataFrame, col: List[str], wavelet='db5', th=0.6) -> pd.DataFrame:
    """
    Wavelet Transform을 적용하는 함수
    Args:
        df (pd.DataFrame): 데이터를 포함한 데이터프레임
        col (str): Wavelet Transform을 적용할 열 이름
        wavelet (str): 사용할 웨이블릿 유형. 기본값은 'db5'
        th (float): 임계값 계수. 기본값은 0.6
    Returns:
        np.ndarray: Wavelet Transform이 적용된 신호
    """
    signal = df[col].values
    th = th*np.nanmax(signal)
    coef = pywt.wavedec(signal, wavelet, mode="per" )
    coef[1:] = (pywt.threshold(i, value=th, mode="soft" ) for i in coef[1:])
    reconstructed = pywt.waverec(coef, wavelet, mode="per" )
    return reconstructed


# 날짜 관련 생성
def make_date(df: pd.DataFrame, date_column: str) -> Tuple[pd.DataFrame]:
    """
    입력된 데이터프레임의 특정 날짜 열을 기준으로 연도, 월, 주, 요일, 시간을 추출하여 새로운 피처로 추가.
    Args:
        df (pd.DataFrame): 날짜 컬럼을 포함하는 데이터프레임.
        date_column (str): 날짜 정보가 담긴 열 이름.
    Returns:
        pd.DataFrame: 날짜 관련 피처가 추가된 데이터프레임 반환.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df['year'] = df[date_column].dt.year  # 연도
    df['month'] = df[date_column].dt.month  # 월 
    df['week'] = df[date_column].dt.isocalendar().week  # 주
    df['day_of_week'] = df[date_column].dt.dayofweek  # 요일 
    df['hour'] = df[date_column].dt.hour  # 시간 
    date_columns = [date_column, 'year', 'month', 'week', 'day_of_week', 'hour']
    return df


# 변동성, 차분 피처 생성 함수 
def make_diff_change(
    df: pd.DataFrame, 
    columns_list: List[str]
) -> pd.DataFrame:
    """
    주어진 변수들에 대한 변동성, 차분 피처를 생성하고, 결측값을 처리하는 함수.
    Args:
        df (pd.DataFrame): 데이터를 포함한 데이터프레임.
        columns_list (List[str]): 변동성과 차분 피처를 추가하고자 하는 열 이름 목록.
    Returns:
        pd.DataFrame: 변동성과 차분 피처가 추가된 데이터프레임.
    """  
    new_features = {}  # 새로 추가할 열을 저장할 딕셔너리
    for col in columns_list:
        if col in df.columns:
            pct_change_col = f'{col}_pct_change'
            diff_col = f'{col}_diff'
            new_features[pct_change_col] = df[col].pct_change(fill_method=None)
            new_features[diff_col] = df[col].diff()
        else:
            print(f"Error: Cannot find '{col}' column")
    new_features_df = pd.DataFrame(new_features, index=df.index) # 새 피처를 데이터프레임으로
    new_features_df = new_features_df.ffill().bfill()  # 결측값 처리 (고정: ffill, bfill)
    df = pd.concat([df, new_features_df], axis=1)  # 기존 데이터프레임에 추가
    return df


# 롱/숏 비율
def make_longshort_ratio(
    df: pd.DataFrame, 
    long_col: str, 
    short_col: str
) -> pd.DataFrame:
    """
    롱/숏 비율을 계산하는 함수
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        long_col (str): 롱(liquidations) 데이터가 저장된 열 이름
        short_col (str): 숏(liquidations) 데이터가 저장된 열 이름
    Returns:
        pd.DataFrame: 롱/숏 비율이 저장된 새로운 열이 추가된 데이터프레임
    """
    df['long_short_ratio'] = df[long_col] / (df[short_col] + 1e-6)  # 분모가 0이 되는 것을 방지하기 위해 작은 값을 더함
    return df


# 청산/거래량 비율
def make_liquidation_to_volume_ratio(
    df: pd.DataFrame, 
    long_col: str, 
    short_col: str, 
    buy_volume_col: str, 
    sell_volume_col: str
) -> pd.DataFrame:
    """
    청산/거래량 비율을 계산하는 함수
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        long_col (str): 롱(liquidations) 데이터가 저장된 열 이름
        short_col (str): 숏(liquidations) 데이터가 저장된 열 이름
        buy_volume_col (str): 매수 거래량이 저장된 열 이름
        sell_volume_col (str): 매도 거래량이 저장된 열 이름
    Returns:
        pd.DataFrame: 청산/거래량 비율을 저장한 새로운 열이 추가된 데이터프레임
    """
    df['liquidation_to_volume_ratio'] = (
        (df[long_col] + df[short_col]) / 
        (df[buy_volume_col] + df[sell_volume_col] + 1e-6)  # 분모가 0이 되는 것을 방지하기 위해 작은 값을 더함
    )
    return df


# 청산된 USD 롱/숏 비율
def make_liquidation_usd_ratio(
    df: pd.DataFrame, 
    long_usd_col: str, 
    short_usd_col: str
) -> pd.DataFrame:
    """
    청산된 USD 롱/숏 비율을 계산하는 함수
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        long_usd_col (str): 롱(liquidations) USD 데이터가 저장된 열 이름
        short_usd_col (str): 숏(liquidations) USD 데이터가 저장된 열 이름 
    Returns:
        pd.DataFrame: 청산된 USD 롱/숏 비율을 저장한 새로운 열이 추가된 데이터프레임
    """
    df['liquidation_usd_ratio'] = df[long_usd_col] / (df[short_usd_col] + 1e-6)  # 분모가 0이 되는 것을 방지하기 위해 작은 값을 더함
    return df


# 펀딩 비율과 롱/숏 포지션 차이 곱
def make_funding_rate_position_change(
    df: pd.DataFrame, 
    funding_rate_col: str, 
    long_liquidations_col: str, 
    short_liquidations_col: str
) -> pd.DataFrame:
    """
    펀딩 비율과 롱/숏 포지션 차이를 곱하여 포지션 변화를 계산하는 함수
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        funding_rate_col (str): 펀딩 비율 데이터가 저장된 열 이름
        long_liquidations_col (str): 롱 청산 데이터가 저장된 열 이름
        short_liquidations_col (str): 숏 청산 데이터가 저장된 열 이름
    Returns:
        pd.DataFrame: 펀딩 비율에 따른 포지션 변화를 저장한 새로운 열이 추가된 데이터프레임
    """
    df['funding_rate_position_change'] = df[funding_rate_col] * (
        df[long_liquidations_col] - df[short_liquidations_col]
    )
    return df


# 프리미엄 갭과 프리미엄 인덱스의 차이
def make_premium_diff(
    df: pd.DataFrame, 
    premium_gap_col: str, 
    premium_index_col: str
) -> pd.DataFrame:
    """
    프리미엄 갭과 프리미엄 인덱스의 차이를 계산하는 함수
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        premium_gap_col (str): 프리미엄 갭 데이터가 저장된 열 이름
        premium_index_col (str): 프리미엄 인덱스 데이터가 저장된 열 이름
    Returns:
        pd.DataFrame: 프리미엄 갭과 프리미엄 인덱스의 차이를 저장한 새로운 열이 추가된 데이터프레임
    """
    df['premium_diff'] = df[premium_gap_col] - df[premium_index_col]
    return df


# 해시레이트와 난이도 간의 비율
def make_hashrate_to_difficulty(
    df: pd.DataFrame, 
    hashrate_col: str, 
    difficulty_col: str
) -> pd.DataFrame:
    """
    해시레이트와 난이도 간의 비율을 계산하는 함수
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        hashrate_col (str): 해시레이트 데이터가 저장된 열 이름
        difficulty_col (str): 난이도 데이터가 저장된 열 이름
    Returns:
        pd.DataFrame: 해시레이트와 난이도 비율을 저장한 새로운 열이 추가된 데이터프레임
    """
    df['hashrate_to_difficulty'] = df[hashrate_col] / (df[difficulty_col] + 1e-6)  # 분모가 0이 되는 것을 방지하기 위해 작은 값을 더함
    return df


# 공급 변화율
def make_supply_change_rate(
    df: pd.DataFrame, 
    new_supply_col: str, 
    total_supply_col: str
) -> pd.DataFrame:
    """
    공급 변화율을 계산하는 함수
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        new_supply_col (str): 새로운 공급량 데이터가 저장된 열 이름
        total_supply_col (str): 총 공급량 데이터가 저장된 열 이름
    Returns:
        pd.DataFrame: 공급 변화율을 저장한 새로운 열이 추가된 데이터프레임
    """
    df['supply_change_rate'] = df[new_supply_col] / (df[total_supply_col] + 1e-6)  # 분모가 0이 되는 것을 방지하기 위해 작은 값을 더함
    return df


def make_multiple_rolling(df, exclude_columns_list):
    """
    여러 윈도우 크기(6, 12, 24, 48)로 이동평균 피처를 생성하는 함수
    
    Parameters:
    - df: 입력 데이터프레임
    - exclude_columns_list: 이동평균을 적용하지 않을 열 이름이 담긴 리스트
    
    Returns:
    - 이동평균 피처가 추가된 데이터프레임
    """
    rolling_features = {}

    # 윈도우 크기별로 변수명 리스트 생성
    rolling_columns_dict = {window: [] for window in [6, 12, 24, 48]}

    for col in df.columns:
        if col not in exclude_columns_list:
            for window in [6, 12, 24, 48]:
                new_col_name = f'{col}_rolling_mean_{window}h'
                rolling_features[new_col_name] = df[col].rolling(window=window).mean()
                # 윈도우 크기에 맞는 리스트에 변수명 저장
                rolling_columns_dict[window].append(new_col_name)

    # 새로운 피처를 한 번에 추가
    rolling_features_df = pd.DataFrame(rolling_features, index=df.index)
    df = pd.concat([df, rolling_features_df], axis=1)
    return df

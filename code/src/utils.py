"""
数据处理与后处理工具模块.

本模块提供:

1. **特征工程** — engineer_features_158plus39 / engineer_features_39 / engineer_features
   生成 158 Alpha 因子 + 39 技术指标 + 基本面 + 动量 + 市场宽度

2. **数据集构建** — create_ranking_dataset_vectorized (物化版)
                     + LazyRankingDataset + build_lazy_ranking_dataset (懒加载版)
   将时序股票数据转为 (sequence, target) 训练样本

3. **标签计算** — compute_excess_returns / compute_aux_labels
   生成排序目标 + 方向/波动/收益辅助标签

4. **后处理** — select_top_stocks_with_gate (收益门控选股)
              + optimize_weights (波动率惩罚权重优化)

5. **损失辅助** — NDCGApproxLoss (可微 NDCG 近似)

参考:
    - 懒加载设计: 2026-07-16-工作记录.md §4.4
    - 市场宽度: 2026-07-09-工作记录.md §4
"""

import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════
# 市场宽度特征 (P1改进)
# ══════════════════════════════════════════════════════════

def compute_market_breadth_features(stock_df):
    """
    计算每日市场宽度特征（跨所有股票的截面统计量）。

    这些特征捕捉市场内部结构信息——"300只股票今天是普涨还是分化"，
    比单一的指数涨跌幅提供更丰富的市场状态信号。

    计算的特征:
      - advance_ratio: 涨跌家数比（上涨股票数/总股票数）
      - return_dispersion: 收益离散度（std of returns）
      - return_skew: 收益偏度（正偏=多数上涨 vs 负偏=少数暴跌拖累）
      - volume_change: 放量/缩量信号（当日均量/前5日均量 - 1）
      - up_volume_ratio: 上涨股票成交额占比
      - high_low_ratio: 创20日新高/新低股票比
      - gap_ratio: 跳空高开股票占比

    Args:
        stock_df: 包含 '日期', '股票代码', '涨跌幅', '成交量', '开盘', '收盘', '最高', '最低' 的DataFrame

    Returns:
        DataFrame with columns: 日期 + market_* 特征列，每日一行
    """
    df = stock_df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    dates = sorted(df['日期'].unique())

    breadth_rows = []

    for d in dates:
        day = df[df['日期'] == d]
        if len(day) < 10:
            continue

        row = {'日期': d}

        # 1. 涨跌家数比
        if '涨跌幅' in day.columns:
            rets = day['涨跌幅'].values
            row['market_advance_ratio'] = float((rets > 0).mean())
            row['market_return_dispersion'] = float(np.std(rets))
            row['market_return_skew'] = float(_safe_skew(rets))
            row['market_return_mean'] = float(np.mean(rets))

        # 2. 成交量变化
        if '成交量' in day.columns:
            vol = day['成交量'].values
            row['market_volume_mean'] = float(np.mean(vol))
            # 需要历史数据计算变化，后续填充
            row['market_volume_change'] = 0.0

        # 3. 上涨股票成交额占比
        if '成交额' in day.columns and '涨跌幅' in day.columns:
            up_mask = day['涨跌幅'] > 0
            if up_mask.sum() > 0:
                total_amount = day['成交额'].sum()
                up_amount = day.loc[up_mask, '成交额'].sum()
                row['market_up_amount_ratio'] = float(up_amount / total_amount) if total_amount > 0 else 0.5
            else:
                row['market_up_amount_ratio'] = 0.0

        # 4. 振幅（日内波动）
        if '振幅' in day.columns:
            row['market_amplitude_mean'] = float(day['振幅'].mean())

        # 5. 高开/低开比
        if '开盘' in day.columns and '收盘' in day.columns:
            # 用前一日收盘和当日开盘判断跳空
            pass  # 需要跨日数据，在外部处理

        breadth_rows.append(row)

    breadth_df = pd.DataFrame(breadth_rows)

    if len(breadth_df) > 5:
        # 填充 volume_change: 当日均量 / 前5日均量 - 1
        if 'market_volume_mean' in breadth_df.columns:
            breadth_df['market_volume_change'] = (
                breadth_df['market_volume_mean'] /
                breadth_df['market_volume_mean'].rolling(5, min_periods=1).mean() - 1
            )

    return breadth_df


def _safe_skew(arr):
    """安全计算偏度，处理常数列"""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3 or np.std(arr) < 1e-12:
        return 0.0
    n = len(arr)
    m3 = np.mean((arr - np.mean(arr)) ** 3)
    s3 = np.std(arr) ** 3
    if s3 < 1e-12:
        return 0.0
    # 调整偏度（小样本修正）
    skew = (n / ((n - 1) * (n - 2))) * n * m3 / s3 if n > 2 else 0.0
    return float(np.clip(skew, -5, 5))


def merge_market_breadth(stock_df, breadth_df):
    """
    将市场宽度特征按日期拼接到每只股票的日特征中。

    这样每只股票在同一天拥有相同的市场宽度值，
    模型可以通过这些特征感知整体市场环境。

    Args:
        stock_df: 股票级别DataFrame
        breadth_df: 日期级别市场宽度DataFrame

    Returns:
        拼接后的DataFrame
    """
    if breadth_df is None or len(breadth_df) == 0:
        return stock_df

    stock_df = stock_df.copy()
    stock_df['日期'] = pd.to_datetime(stock_df['日期'])
    breadth_df['日期'] = pd.to_datetime(breadth_df['日期'])

    breadth_cols = [c for c in breadth_df.columns if c != '日期']
    merged = pd.merge(stock_df, breadth_df, on='日期', how='left')

    # 前向填充节假日缺失值
    for col in breadth_cols:
        merged[col] = merged[col].ffill().fillna(0)

    print(f"  拼接市场宽度特征: {len(breadth_cols)} 维")
    return merged


# 特征工程
def _rolling_linear_regression(x, y):
    x = np.vstack([np.ones(len(x)), x]).T
    beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return beta[1], res[0] if len(res) > 0 else 0.0, np.sum((y - (x @ beta))**2)
def engineer_features_158plus39(df):
    """
    计算39个技术指标特征和158个Alpha特征，并合并它们。
    """
    # 为了避免修改原始DataFrame，创建一个副本
    df_copy = df.copy()

    # 1. 计算158个Alpha特征
    df_158 = engineer_features(df_copy)
    
    # 2. 计算39个技术指标特征
    df_39 = engineer_features_39(df_copy)

    # 3. 合并两个DataFrame
    # 首先，从df_39中选取我们需要的列，避免与df_158中的原始列（如'开盘'）重复
    feature_cols_39 = [
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 
        'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 
        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 
        'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
        'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    ]
    
    # 确保所有列都存在于df_39中
    feature_cols_39_exist = [col for col in feature_cols_39 if col in df_39.columns]
    
    # 合并，df_158 已经包含了原始列和158个特征
    df_final = pd.concat([df_158, df_39[feature_cols_39_exist]], axis=1)

    # 4. 处理可能因为合并产生的重复列（如果两个函数生成了同名特征）
    df_final = df_final.loc[:,~df_final.columns.duplicated()]

    # 5. 统一处理inf和NaN
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(0, inplace=True)
    
    return df_final

def engineer_features_39(df):
    """
    计算39个技术指标特征。
    'stock_idx','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
    'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 
    'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  
    'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
    """
    try:
        import talib
        import numpy as np
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    # 移动平均线 (SMA, EMA)
    df['sma_5'] = talib.SMA(close, timeperiod=5)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)
    df['ema_60'] = talib.EMA(close, timeperiod=60)

    # MACD
    macd_line, macd_signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_line

    # RSI
    df['rsi'] = talib.RSI(close, timeperiod=14)

    # KDJ
    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    # Bollinger Bands
    df['boll_mid'], df['boll_upper'], df['boll_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    # 标准差 = (上轨 - 中轨) / 2
    df['boll_std'] = (df['boll_upper'] - df['boll_mid']) / 2

    # 删除临时列
    df.drop(columns=['boll_upper', 'boll_lower'], inplace=True)

    # ATR
    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    # OBV (On-Balance Volume)
    df['obv'] = talib.OBV(close, volume)

    # Volume-related features
    df['volume_change'] = volume.pct_change()
    df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
    df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']

    # Returns and Volatility
    df['return_1'] = close.pct_change(1)
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    # Spreads
    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    # 处理 inf 和 -inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 填充 NaN 值（注意：这可能引入偏差，根据下游任务决定是否保留）
    df.fillna(0, inplace=True)

    return df

def engineer_features(df):
    """
    使用talib加速特征计算
    """
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    # 为了避免修改原始DataFrame，创建一个副本
    df = df.copy()

    # 数据不足时返回空特征（NaN），后续会被 fillna(0) 处理
    if len(df) < 5:
        result = df[['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额',
                     '振幅', '涨跌额', '换手率', '涨跌幅']].copy()
        return result

    # 基础变量
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    # 特征列表
    features = []
    feature_names = []

    # 1. K-line features (9 features) - 向量化操作，速度很快，无需更改
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12)
    ])
    feature_names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

    # 2. Price-related features (4 features) - 向量化操作，无需更改
    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    windows = [5, 10, 20, 30, 60]

    # 3. Price change features (5 features) - 向量化操作，无需更改
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    # 4. Moving average features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    # 5. Standard deviation features (5 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    # 6. Regression-based features (15 features) - 使用 talib 加速
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')
        
        # R-squared can be calculated as CORREL^2
        if len(close) >= w:
            time_period_series = pd.Series(range(w), index=close.index[:w])
            rolling_corr = close.rolling(w).corr(time_period_series)
            rsquare = rolling_corr**2
            features.append(rsquare)
        else:
            features.append(pd.Series(np.full(len(close), np.nan), index=close.index))
        feature_names.append(f'RSQR{w}')

        # Residuals
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        resi = close - predicted
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    # 7. Max/Min features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    # 8. Quantile features (10 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    # 9. Rank features (5 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    # 10. Stochastic oscillator features (5 features) - talib.STOCH 计算的是另一指标，保留原实现
    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f'RSV{w}')

    # 11. Index of Max/Min features (15 features) - talib 不支持，保留原实现
    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f'IMAX{w}')
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f'IMIN{w}')
    for w in windows:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        feature_names.append(f'IMXD{w}')

    # 12. Correlation features (10 features) - 使用 talib 加速
    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f'CORR{w}')
    
    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        # talib.CORREL 需要 Series，且不能有 NaN
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f'CORD{w}')

    # 13. Count features (15 features) - 向量化操作，无需更改
    close_diff_pos = (close > close.shift(1))
    close_diff_neg = (close < close.shift(1))
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f'CNTP{w}')
    for w in windows:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f'CNTN{w}')
    for w in windows:
        cntp = close_diff_pos.rolling(w).mean()
        cntn = close_diff_neg.rolling(w).mean()
        features.append(cntp - cntn)
        feature_names.append(f'CNTD{w}')

    # 14. Sum of price change features (15 features) - 向量化操作，无需更改
    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'SUMP{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'SUMN{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'SUMD{w}')

    # 15. Volume-related features (10 features) - 使用 talib 加速
    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    # 16. Weighted volume features (5 features) - 向量化操作，无需更改
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vol_w_ret = vol_weighted_ret.rolling(w).mean()
        std_vol_w_ret = vol_weighted_ret.rolling(w).std()
        features.append(std_vol_w_ret / (mean_vol_w_ret + 1e-12))
        feature_names.append(f'WVMA{w}')

    # 17. Volume change sum features (15 features) - 向量化操作，无需更改
    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'VSUMP{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'VSUMN{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'VSUMD{w}')

    # Combine all features into a new DataFrame
    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names
    
    # Merge with original df
    df = pd.concat([df, feature_df], axis=1)
    
    # 填充缺失值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df
def process_single_stock(stock_row, data, features, sequence_length, date):
    """处理单只股票的数据，返回序列、目标值和股票索引"""
    stock_code = stock_row['instrument']
    # stock_idx = stock_row['stock_idx']
    
    # 获取该股票历史sequence_length天的数据（包括当天）
    stock_history = data[
        (data['instrument'] == stock_code) & 
        (data['datetime'] <= date)
    ].sort_values('datetime').tail(sequence_length)

    if len(stock_history) == sequence_length:
        seq = stock_history[features].values
        target = stock_row['label']  # 下一天的涨跌幅
        return seq, target, stock_code
    else:
        return None, None, None

def process_single_date(date, data, features, sequence_length):
    """处理单个日期的所有股票数据"""
    try:
        # 获取当天有target的股票（即有下一天数据的股票）
        day_data = data[data['datetime'] == date]
        day_data = day_data.dropna(subset=['label'])  # 确保有target
        
        if len(day_data) < 10:  # 确保至少有10只股票
            return None
            
        # 获取当天所有股票的特征序列
        day_sequences = []
        day_targets = []
        day_stock_indices = []
        
        # 对于单个日期内的股票处理，仍使用串行方式避免过度并行化
        # 因为多进程的开销可能超过收益
        for _, stock_row in day_data.iterrows():
            seq, target, stock_idx = process_single_stock(
                stock_row, data, features, sequence_length, date
            )
            if seq is not None:
                day_sequences.append(seq)
                day_targets.append(target)
                day_stock_indices.append(stock_idx)
        
        if len(day_sequences) >= 10:  # 确保有足够的股票
            # 创建排序标签：涨跌幅越高，相关性得分越高
            day_targets = np.array(day_targets)
            # 使用涨跌幅的排序作为相关性得分（值越大排名越高）
            sorted_indices = np.argsort(day_targets)[::-1]  # 降序排列
            relevance = np.zeros_like(day_targets, dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(day_targets) - rank  # 最高涨跌幅得分最高
            
            return {
                'sequences': np.array(day_sequences),
                'targets': day_targets,
                'relevance': relevance,
                'stock_indices': day_stock_indices,
                'date': date
            }
        else:
            return None
            
    except Exception as e:
        print(f"处理日期 {date} 时出错: {e}")
        return None

def create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path=None, max_workers=None):
    """
    输入：股票历史数据 DataFrame，特征列名列表，序列长度，排名数据保存路径，最大工作进程数
    输出：排序数据集，格式为：(sequences, targets, relevance_scores, stock_indices)
    - sequences: List of np.array, 每个元素形状为 (num_stocks, sequence_length, num_features)
    - targets: List of np.array, 每个元素形状为 (num_stocks,)
    - relevance_scores: List of np.array, 每个元素形状为 (num_stocks,)
    - stock_indices: List of List, 每个元素为对应股票的索引列表
    """
    """多进程版本的排序数据集创建函数"""
    if ranking_data_path is not None:
        # 如果指定了ranking_data_path，尝试加载已有的数据集
        if os.path.exists(ranking_data_path):
            print(f"加载已有的排序数据集: {ranking_data_path}")
            return joblib.load(ranking_data_path)
    """
    创建排序数据集，按日期组织数据，每个样本包含同一天所有股票的特征和涨跌幅排序
    使用多线程加速处理
    """
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    
    print("正在创建排序数据集（多线程版本）...")
    
    # 获取所有日期，确保有足够的历史数据
    all_dates = sorted(data['datetime'].unique())
    min_date_for_sequences = all_dates[sequence_length-1]  # 确保有足够历史数据
    
    # 只使用有足够历史数据的日期
    valid_dates = [date for date in all_dates if date >= min_date_for_sequences]
    
    print(f"总日期数: {len(all_dates)}, 有效日期数: {len(valid_dates)}")
    
    # 设置最大工作进程数
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from tqdm import tqdm
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 10)
    
    print(f"使用 {max_workers} 个进程处理数据")
    
    # 分批处理日期以避免内存问题
    processed_count = 0
        
    # 使用进程池并行处理日期批次
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 创建处理函数的偏函数
            process_func = partial(process_single_date,
                                    data=data,
                                    features=features,
                                    sequence_length=sequence_length)
            
            # 并行处理批次中的所有日期
            futures = [executor.submit(process_func, date) for date in valid_dates]
            
            # 收集结果
            for future in tqdm(futures, desc="Processing dates", total=len(valid_dates)):
                try:
                    result = future.result(timeout=60)  # 设置超时
                    if result is not None:
                        sequences.append(result['sequences'])
                        targets.append(result['targets'])
                        relevance_scores.append(result['relevance'])
                        stock_indices.append(result['stock_indices'])
                        processed_count += 1
                except Exception as e:
                    print(f"处理某个日期时出错: {e}")
                    continue
                    
    except Exception as e:
        print(f"进程池处理出错，回退到串行处理: {e}")
        # 如果多进程出错，回退到串行处理
        for date in tqdm(valid_dates, desc="串行处理"):
            result = process_single_date(date, data, features, sequence_length)
            if result is not None:
                sequences.append(result['sequences'])
                targets.append(result['targets'])
                relevance_scores.append(result['relevance'])
                stock_indices.append(result['stock_indices'])
                processed_count += 1
    
    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        print(f"每个样本平均包含 {np.mean([len(seq) for seq in sequences]):.1f} 只股票")
    
    # 将四个数据保存下来，下次直接读取
    if ranking_data_path:
        joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
        print(f"数据集已保存到: {ranking_data_path}")
    
    return sequences, targets, relevance_scores, stock_indices

def create_dataset(data, features, sequence_length, ranking_data_path=None):
    """保持原有接口，但内部调用新的排序数据集创建函数"""
    return create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path)

def create_ranking_dataset_vectorized(data, features, sequence_length, ranking_data_path=None, min_window_end_date=None,
                                       max_future_span_days=None):
    """
    向量化加速版本：预计算每只股票的所有滑动窗口，再按日期聚合。
    保持与原函数完全相同的输出格式。

    Args:
        max_future_span_days: 未来5条数据的自然日跨度上限。
            None（默认）= 严格模式：要求未来5天自然日连续（等价于只保留周五预测日，
            与官方测试窗口口径一致，但会丢弃约80%样本）。
            数值（如15）= 放宽模式：允许跨周末/节假日，仅过滤长期停牌，
            训练样本量约提升5倍。
    """
    # if ranking_data_path and os.path.exists(ranking_data_path):
    #     print(f"加载已有的排序数据集: {ranking_data_path}")
    #     return joblib.load(ranking_data_path)

    print("正在创建排序数据集（向量化加速版本）...")
    # data.rename(columns={'stock_idx': 'instrument'}, inplace=True)
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    # 1. 确保数据按股票和时间排序
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    # 2. 确保每只股票都有 'label'（次日涨跌幅），否则无法作为 target
    data = data.dropna(subset=['label'])
    
    # 3. 为每只股票生成所有滑动窗口
    # 仅保留满足以下条件的 end_date：
    # - 历史窗口长度满足 sequence_length
    # - end_date 之后存在 5 条未来数据
    # - 严格模式(默认)：5 条未来数据自然日连续；放宽模式：自然日跨度 ≤ max_future_span_days
    all_windows = []  # 每个元素: (end_date, stock_code, sequence, target, target_abs, market_label)

    # 检查是否有绝对收益标签和市场标签
    has_label_abs = 'label_abs' in data.columns
    has_market_label = 'market_label' in data.columns

    print("Step 1: 为每只股票生成滑动窗口...")
    grouped = data.groupby('instrument')

    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue

        # 提取特征和 label
        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)           # (T,) 排序标签
        labels_abs = group['label_abs'].values.astype(np.float32) if has_label_abs else labels  # (T,) 绝对收益
        market_labels = group['market_label'].values.astype(np.float32) if has_market_label else np.zeros_like(labels)  # (T,) 市场方向
        dates = group['datetime'].values                            # (T,)
        dates_day = group['datetime'].values.astype('datetime64[D]')

        # 生成滑动窗口：从第 sequence_length-1 行开始（0-indexed）
        num_windows = len(group) - sequence_length + 1
        n = len(group)
        for i in range(num_windows):
            end_idx = i + sequence_length - 1

            # 需要有未来 5 条数据
            if end_idx + 5 >= n:
                continue

            # 未来 5 条数据的日期约束
            future_dates = dates_day[end_idx + 1:end_idx + 6]
            if max_future_span_days is None:
                # 严格模式：自然日连续（只保留完整交易周的预测日）
                future_diffs = np.diff(future_dates).astype(np.int64)
                if not np.all(future_diffs == 1):
                    continue
            else:
                # 放宽模式：允许跨周末/节假日，仅过滤长期停牌导致的异常跨度
                span = (future_dates[-1] - future_dates[0]).astype(np.int64)
                if span > max_future_span_days:
                    continue

            seq = feature_values[i : i + sequence_length]   # (L, F)
            target = labels[end_idx]                        # 排序标签
            target_abs = labels_abs[end_idx]                # 绝对收益（用于 return_head 回归）
            market_lbl = market_labels[end_idx]             # 市场方向标签（★ 市场聚合）
            end_date = dates[end_idx]                       # 窗口结束日期（即预测日）
            all_windows.append((end_date, stock_code, seq, target, target_abs, market_lbl))

    # 4. 转为 DataFrame 便于按日期聚合
    print("Step 2: 按日期聚合窗口...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'seq', 'target', 'target_abs', 'market_label'])

    # 5. 按 date 分组，构建每日样本
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    aux_labels = []  # 包含 direction, volatility, return_abs

    print("Step 3: 构建每日样本并计算 relevance...")
    grouped_by_date = window_df.groupby('date')

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)

    for date, group in tqdm(grouped_by_date, desc="Aggregating by date"):
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue

        if len(group) < 10:
            continue

        # 提取数据
        day_seqs = np.stack(group['seq'].values)          # (N, L, F)
        day_targets = group['target'].values              # (N,) 超额收益
        day_targets_abs = group['target_abs'].values      # (N,) 绝对收益
        day_stocks = group['stock_code'].tolist()         # [str]

        # 计算 relevance（与原逻辑一致，基于超额收益排序）
        sorted_indices = np.argsort(day_targets)[::-1]
        relevance = np.zeros_like(day_targets, dtype=np.float32)
        for rank, idx in enumerate(sorted_indices):
            relevance[idx] = len(day_targets) - rank

        # 构建辅助标签（含绝对收益 + 市场方向）
        day_market_labels = group['market_label'].values  # (N,) 市场方向
        day_aux = {
            'direction': (day_targets > 0).astype(np.float32),
            'volatility': np.abs(day_targets).astype(np.float32),
            'return_abs': day_targets_abs.astype(np.float32),  # 绝对收益标签
            'market_label': day_market_labels.astype(np.float32),  # ★ 市场方向标签
        }

        sequences.append(day_seqs)
        targets.append(day_targets)
        relevance_scores.append(relevance)
        stock_indices.append(day_stocks)
        aux_labels.append(day_aux)

    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        avg_stocks = np.mean([len(seq) for seq in sequences])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

    # 6. 保存
    # if ranking_data_path:
    #     joblib.dump((sequences, targets, relevance_scores, stock_indices, aux_labels), ranking_data_path)
    #     print(f"数据集已保存到: {ranking_data_path}")

    return sequences, targets, relevance_scores, stock_indices, aux_labels


# ══════════════════════════════════════════════════════════
# 辅助标签构建：超额收益 + 方向 + 波动率
# ══════════════════════════════════════════════════════════

def compute_excess_returns(processed_df):
    """
    将绝对收益标签转为超额收益（相对当日所有股票均值）。

    三队获奖经验均指出：绝对收益受大盘情绪干扰严重，
    转换为超额收益后模型聚焦"谁比谁好"的相对排序。

    Args:
        processed_df: 包含 '日期' 和 'label'（绝对收益）的 DataFrame

    Returns:
        添加了 'label_raw' 列，并将 'label' 替换为超额收益的 DataFrame
    """
    processed_df = processed_df.copy()
    # 保存原始绝对收益
    processed_df['label_raw'] = processed_df['label'].copy()
    # 按日期计算市场均值
    day_mean = processed_df.groupby('日期')['label'].transform('mean')
    # 超额收益
    processed_df['label'] = processed_df['label'] - day_mean
    return processed_df


def compute_aux_labels(processed_df):
    """
    为辅助任务构建标签。

    - direction: label > 0 为正向（1），否则为负向（0）
    - volatility: |label| 作为波动率标签

    Args:
        processed_df: 包含 'label' 列的 DataFrame

    Returns:
        添加了 'direction' 和 'volatility' 列的 DataFrame
    """
    processed_df = processed_df.copy()
    # 方向标签（二分类）
    processed_df['direction'] = (processed_df['label'] > 0).astype(np.float32)
    # 波动率标签（收益的绝对值）
    processed_df['volatility'] = np.abs(processed_df['label']).astype(np.float32)
    return processed_df


# ══════════════════════════════════════════════════════════
# LambdaRank / NDCG 损失
# ══════════════════════════════════════════════════════════

class NDCGApproxLoss(nn.Module):
    """
    NDCG 近似的可微排序损失。

    支持两种近似模式:
    - softmax: 标准 softmax 近似（平滑但模糊）
    - gumbel:  Gumbel-Softmax 近似（更尖锐，对 Top-K 更敏感）

    借鉴 LambdaRank 思想：排名越靠前的样本对 NDCG 贡献越大，
    通过 DCG 的折扣因子 log2(rank+1) 实现位置敏感加权。

    与 WeightedRankingLoss 组合使用，让模型更关注 Top 位置的排序质量。
    """

    def __init__(self, k=5, temperature=1.0, use_gumbel=False, gumbel_hard=False):
        super(NDCGApproxLoss, self).__init__()
        self.k = k
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.gumbel_hard = gumbel_hard

    def _sample_gumbel(self, shape, device):
        """采样 Gumbel(0, 1) 分布"""
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + 1e-20) + 1e-20)

    def _gumbel_softmax(self, logits, temperature, hard=False):
        """
        Gumbel-Softmax 采样。

        相比 softmax:
        - 训练时加入 Gumbel 噪声 → 更尖锐的近似
        - 推理时可设为 hard=True → 精确的 one-hot
        """
        gumbels = self._sample_gumbel(logits.shape, logits.device)
        y = logits + gumbels
        y_soft = F.softmax(y / temperature, dim=-1)

        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
            # Straight-through estimator
            return (y_hard - y_soft).detach() + y_soft
        return y_soft

    def _dcg(self, scores, relevance):
        """计算 DCG: sum(relevance[i] / log2(i+2))"""
        positions = torch.arange(1, len(scores) + 1, device=scores.device, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1.0)
        return (relevance * discounts).sum()

    def _ndcg(self, pred_scores, true_relevance, k=None):
        """
        计算近似 NDCG@k。
        使用 softmax / Gumbel-Softmax 排名近似实现可微性。
        """
        if k is None:
            k = self.k

        device = pred_scores.device
        n = pred_scores.size(0)
        k_actual = min(k, n)

        # 按真实相关性排序得到 ideal DCG
        true_sorted, _ = torch.sort(true_relevance, descending=True)
        ideal_dcg = self._dcg(true_sorted[:k_actual], true_sorted[:k_actual])

        if ideal_dcg < 1e-12:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 用 softmax/Gumbel-Softmax 生成近似排名分布
        scaled_scores = pred_scores / self.temperature

        if self.use_gumbel and self.training:
            pred_prob = self._gumbel_softmax(scaled_scores, temperature=1.0,
                                             hard=self.gumbel_hard)
        else:
            pred_prob = F.softmax(scaled_scores, dim=0)

        # 按预测概率降序排列的近似 DCG
        sorted_probs, sorted_indices = torch.sort(pred_prob, descending=True)
        sorted_rel = true_relevance[sorted_indices]

        positions = torch.arange(1, k_actual + 1, device=device, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1.0)
        approx_dcg = (sorted_rel[:k_actual] * discounts).sum()

        ndcg = approx_dcg / (ideal_dcg + 1e-12)
        return 1.0 - ndcg  # 返回损失（越小越好）

    def forward(self, y_pred, y_true):
        """
        y_pred: [batch, num_items] — 预测分数
        y_true: [batch, num_items] — 真实标签（超额收益）
        """
        batch_size = y_pred.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            total_loss += self._ndcg(y_pred[i], y_true[i])

        return total_loss / batch_size


# ══════════════════════════════════════════════════════════
# 后处理权重优化
# ══════════════════════════════════════════════════════════

def optimize_weights(scores, volatilities=None, top_k=5, candidate_k=10,
                     use_volatility_penalty=True, temperature=2.0):
    """
    分数校准 + 风险调整权重分配。

    借鉴「7355608」的分数融合 + 「O_O」的均值方差思想：
    1. 从候选池 (candidate_k) 中 softmax 归一化分数
    2. 可选：波动率惩罚 → 权重向低波动股票倾斜
    3. 重归一化后取 Top-K

    Args:
        scores: np.array, 预测分数 [num_stocks]
        volatilities: np.array or None, 每只股票的波动率
        top_k: 最终输出股票数
        candidate_k: 候选池大小
        use_volatility_penalty: 是否启用波动率惩罚
        temperature: softmax 温度（越大越平滑）

    Returns:
        top_indices, top_weights
    """
    num_stocks = len(scores)
    k_candidate = min(candidate_k, num_stocks)

    # Step 1: 取 Top-candidate_k 分数
    candidate_indices = np.argsort(scores)[::-1][:k_candidate]
    candidate_scores = scores[candidate_indices]

    # Step 2: softmax 归一化得到初步权重
    # 温度缩放使权重不过于极端
    scaled = candidate_scores / temperature
    scaled = scaled - np.max(scaled)  # 数值稳定
    raw_weights = np.exp(scaled) / np.exp(scaled).sum()

    # Step 3: 波动率惩罚（可选）
    if use_volatility_penalty and volatilities is not None:
        candidate_vols = volatilities[candidate_indices]
        # 高波动 → 高惩罚 → 低权重
        vol_penalty = 1.0 + np.abs(candidate_vols)
        adjusted_weights = raw_weights / vol_penalty
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
    else:
        adjusted_weights = raw_weights

    # Step 4: 按调整后权重排序，取 Top-K
    final_order = np.argsort(adjusted_weights)[::-1]
    top_indices = candidate_indices[final_order[:top_k]]
    top_weights = adjusted_weights[final_order[:top_k]]

    # 最终归一化
    top_weights = top_weights / top_weights.sum()

    return top_indices, top_weights


# ══════════════════════════════════════════════════════════
# 收益门控选股 (Phase 3: 绝对收益过滤 + Sharpe-like评分)
# ══════════════════════════════════════════════════════════

def select_top_stocks_with_gate(scores, predicted_returns=None, volatilities=None,
                                 top_k=5, candidate_k=10, min_return_threshold=0.0,
                                 temperature=2.0, fallback='equal'):
    """
    收益门控选股：只选预测绝对收益为正的股票，用 Sharpe-like 评分排序。

    核心改进:
    1. 过滤: predicted_abs_return < min_return_threshold 的股票不入选
    2. 评分: ranking_score * (1 + predicted_return) / (1 + volatility)
    3. 回退: 如果通过门控的股票不足 top_k，等权选原始 Top-K

    Args:
        scores: np.array [N] 模型排序分数
        predicted_returns: np.array [N] or None, 模型预测的绝对收益率
        volatilities: np.array [N] or None, 波动率
        top_k: 最终输出股票数
        candidate_k: 候选池大小
        min_return_threshold: 最小绝对收益阈值（默认0即只要正收益）
        temperature: softmax 温度
        fallback: 'equal' | 'topk' 回退策略

    Returns:
        top_indices, top_weights
    """
    N = len(scores)
    k_candidate = min(candidate_k, N)

    # Step 1: 取候选池
    candidate_indices = np.argsort(scores)[::-1][:k_candidate]
    candidate_scores = scores[candidate_indices]

    # Step 2: 收益门控过滤（如果有预测绝对收益）
    if predicted_returns is not None:
        candidate_returns = predicted_returns[candidate_indices]
        # 只保留预测正收益的股票
        gate_mask = candidate_returns > min_return_threshold
        n_passed = gate_mask.sum()

        if n_passed >= top_k:
            # 足够多的股票通过门控，缩减候选池
            candidate_indices = candidate_indices[gate_mask]
            candidate_scores = candidate_scores[gate_mask]
        elif n_passed > 0:
            # 通过门控的股票不足 top_k，扩大候选池
            expanded_k = min(candidate_k * 3, N)
            expanded_indices = np.argsort(scores)[::-1][:expanded_k]
            expanded_returns = predicted_returns[expanded_indices]
            expanded_gate = expanded_returns > min_return_threshold
            if expanded_gate.sum() >= top_k:
                candidate_indices = expanded_indices[expanded_gate]
                candidate_scores = scores[candidate_indices]
            else:
                # 扩大后仍然不足，使用通过门控的股票 + 回退
                candidate_indices = expanded_indices[expanded_gate]
                candidate_scores = scores[candidate_indices]
        else:
            # 没有股票通过门控 → 回退策略
            if fallback == 'equal':
                # 等权选原始 Top-K
                top_indices = np.argsort(scores)[::-1][:top_k]
                top_weights = np.ones(top_k) / top_k
                return top_indices, top_weights

    # Step 3: 计算 Sharpe-like 综合评分
    k_cur = len(candidate_indices)
    if k_cur == 0:
        # 安全回退
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_weights = np.ones(top_k) / top_k
        return top_indices, top_weights

    # 基于 ranking score 的综合评分
    combined = candidate_scores.copy().astype(np.float64)

    # 融入预测绝对收益（正收益加分，负收益减分）
    if predicted_returns is not None:
        cur_returns = predicted_returns[candidate_indices]
        # 收益加权: score *= (1 + return)  正收益放大，负收益缩小
        return_factor = 1.0 + cur_returns
        return_factor = np.clip(return_factor, 0.1, 5.0)  # 防止极端值
        combined = combined * return_factor

    # 波动率惩罚（适度，不主导评分）
    if volatilities is not None:
        cur_vols = volatilities[candidate_indices]
        # 使用温和的惩罚: score /= (1 + vol)
        vol_penalty = 1.0 + np.abs(cur_vols)
        combined = combined / vol_penalty

    # Step 4: Softmax 计算权重，取 Top-K
    scaled = combined / temperature
    scaled = scaled - np.max(scaled)  # 数值稳定
    weights = np.exp(scaled) / np.exp(scaled).sum()

    # 按权重排序取 Top-K
    final_order = np.argsort(weights)[::-1]
    top_indices = candidate_indices[final_order[:top_k]]
    top_weights = weights[final_order[:top_k]]

    # 最终归一化
    top_weights = top_weights / top_weights.sum()

    return top_indices, top_weights


# ══════════════════════════════════════════════════════════
# 懒加载排序数据集（长历史训练的内存安全方案）
# ══════════════════════════════════════════════════════════

class LazyRankingDataset(torch.utils.data.Dataset):
    """
    内存安全的排序数据集：每只股票的特征矩阵只存一份（float32），
    60 天窗口切片延迟到 __getitem__ 执行。

    为什么需要：物化版本（create_ranking_dataset_vectorized + RankingDataset）
    每个"天样本"存一份 (300股, 60天, 237特征) 数组 ≈ 17MB，2010 年起的
    训练窗口约 3400 个天样本 ≈ 58GB，远超内存；懒加载只需 ~1GB。

    __getitem__ 返回的字典与 RankingDataset 完全一致，兼容现有 collate_fn。
    """

    def __init__(self, stock_store, day_samples):
        """
        Args:
            stock_store: {instrument: {'features': (T,F) float32 ndarray}}
            day_samples: list of dict，每天一个样本：
                {'stocks': [(instrument, end_idx)], 'targets': (N,) float32,
                 'relevance': (N,) float32, 'stock_indices': (N,) int64,
                 'aux': {'direction','volatility','return_abs','market_label'}}
        """
        self.stock_store = stock_store
        self.day_samples = day_samples
        # 序列长度由构建函数写入
        self.sequence_length = None

    def __len__(self):
        return len(self.day_samples)

    @property
    def sample_dates(self):
        """每个天样本的预测日，用于时间衰减采样权重计算"""
        return [d['date'] for d in self.day_samples]

    def __getitem__(self, idx):
        day = self.day_samples[idx]
        L = self.sequence_length
        seqs = np.stack([
            self.stock_store[inst]['features'][end_idx - L + 1: end_idx + 1]
            for inst, end_idx in day['stocks']
        ])  # (N, L, F)

        item = {
            'sequences': torch.from_numpy(seqs),
            'targets': torch.from_numpy(day['targets']),
            'relevance': torch.LongTensor(day['relevance']),
            'stock_indices': torch.LongTensor(day['stock_indices']),
            'direction': torch.from_numpy(day['aux']['direction']),
            'volatility': torch.from_numpy(day['aux']['volatility']),
            'return_abs': torch.from_numpy(day['aux']['return_abs']),
            'market_label': torch.from_numpy(day['aux']['market_label']),
        }
        return item


def build_lazy_ranking_dataset(data, features, sequence_length, min_window_end_date=None,
                               max_future_span_days=None):
    """
    构建 LazyRankingDataset。窗口有效性过滤逻辑与
    create_ranking_dataset_vectorized 完全一致（含 max_future_span_days 语义），
    relevance / 辅助标签的计算方式也一致，仅存储方式不同。
    """
    print("正在创建排序数据集（懒加载版本）...")
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    data = data.dropna(subset=['label'])

    has_label_abs = 'label_abs' in data.columns
    has_market_label = 'market_label' in data.columns

    stock_store = {}
    window_records = []  # (end_date, instrument, end_idx, target, target_abs, market_label)

    print("Step 1: 缓存每只股票的特征矩阵并索引有效窗口...")
    for stock_code, group in tqdm(data.groupby('instrument'), desc="Processing stocks"):
        if len(group) < sequence_length:
            continue

        feature_values = group[features].values.astype(np.float32)
        labels = group['label'].values.astype(np.float32)
        labels_abs = group['label_abs'].values.astype(np.float32) if has_label_abs else labels
        market_labels = (group['market_label'].values.astype(np.float32)
                         if has_market_label else np.zeros_like(labels))
        dates = group['datetime'].values
        dates_day = group['datetime'].values.astype('datetime64[D]')

        stock_store[stock_code] = {'features': feature_values}

        n = len(group)
        for i in range(n - sequence_length + 1):
            end_idx = i + sequence_length - 1
            if end_idx + 5 >= n:
                continue
            future_dates = dates_day[end_idx + 1:end_idx + 6]
            if max_future_span_days is None:
                future_diffs = np.diff(future_dates).astype(np.int64)
                if not np.all(future_diffs == 1):
                    continue
            else:
                span = (future_dates[-1] - future_dates[0]).astype(np.int64)
                if span > max_future_span_days:
                    continue
            window_records.append((dates[end_idx], stock_code, end_idx,
                                   labels[end_idx], labels_abs[end_idx], market_labels[end_idx]))

    print("Step 2: 按日期聚合并预计算标签...")
    window_df = pd.DataFrame(window_records,
                             columns=['date', 'instrument', 'end_idx',
                                      'target', 'target_abs', 'market_label'])
    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)

    day_samples = []
    for date, group in tqdm(window_df.groupby('date'), desc="Aggregating by date"):
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue
        if len(group) < 10:
            continue

        day_targets = group['target'].values.astype(np.float32)
        # relevance 与物化版本一致：按 target 降序，rank 0 得分 N
        sorted_indices = np.argsort(day_targets)[::-1]
        relevance = np.zeros_like(day_targets, dtype=np.float32)
        for rank, idx in enumerate(sorted_indices):
            relevance[idx] = len(day_targets) - rank

        day_samples.append({
            'date': date,
            'stocks': list(zip(group['instrument'].tolist(), group['end_idx'].tolist())),
            'targets': day_targets,
            'relevance': relevance,
            'stock_indices': np.array(group['instrument'].values, dtype=np.int64),  # copy 保证可写
            'aux': {
                'direction': (day_targets > 0).astype(np.float32),
                'volatility': np.abs(day_targets).astype(np.float32),
                'return_abs': group['target_abs'].values.astype(np.float32),
                'market_label': group['market_label'].values.astype(np.float32),
            },
        })

    dataset = LazyRankingDataset(stock_store, day_samples)
    dataset.sequence_length = sequence_length

    feat_bytes = sum(v['features'].nbytes for v in stock_store.values())
    print(f"成功创建 {len(day_samples)} 个训练样本（懒加载，特征缓存 {feat_bytes/1e9:.2f} GB）")
    if day_samples:
        avg_stocks = np.mean([len(d['stocks']) for d in day_samples])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")
    return dataset
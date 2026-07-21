"""
训练脚本 — 沪深300排序模型训练主入口.

本模块包含完整的训练流水线:

1. **标签构建** — _build_label_and_clean: 混合标签 (70% 排序 + 30% 绝对收益)
2. **数据工程** — preprocess_data: 特征选择 + 标准化 + 标签构建
3. **损失函数** — WeightedRankingLoss: Listwise + Pairwise + NDCG + Precision@K
                         + Portfolio Return Loss (Gumbel-Softmax Top-K)
                         + Market Direction BCE
4. **训练循环** — train_ranking_model: 带早停的单窗口训练
5. **评估** — evaluate_ranking_model: 验证集 final_score 计算
6. **预测** — predict_top_stocks: 对未来日期做 Top-K 预测

主要入口:
    python code/src/train.py              # 单窗口训练 (默认配置)
    python code/src/train.py --seed 42    # 指定随机种子

参考:
    - 损失设计: 改进方案.md §2.2 (Portfolio-Level 损失)
    - 标签设计: 改进方案.md §2.1 (混合标签)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized
from utils import compute_excess_returns, compute_aux_labels, NDCGApproxLoss, optimize_weights
from fundamental import (load_fundamentals, engineer_fundamental_features,
                         engineer_features_158plus39_fundamental, engineer_features_all,
                         FUNDAMENTAL_FEATURE_COLS, MOMENTUM_FEATURE_COLS)
import joblib
import os
import sys
import json
import multiprocessing as mp
import random
import math

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # 集成训练不需要 bit-exact
    torch.backends.cudnn.benchmark = True        # 加速训练
    os.environ['PYTHONHASHSEED'] = str(seed)

feature_cloums_map = {
    '39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39+fundamental': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'] + FUNDAMENTAL_FEATURE_COLS,

    '158+39+fundamental+momentum': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'] + FUNDAMENTAL_FEATURE_COLS + MOMENTUM_FEATURE_COLS,
}
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39,
    '158+39+fundamental': engineer_features_158plus39_fundamental,
    '158+39+fundamental+momentum': engineer_features_all,
}


def _build_label_and_clean(processed, drop_small_open=True, use_quantile_label=False,
                           use_mixed_label=False, mixed_alpha=0.7):
    """统一构建标签并清洗无效样本。

    Args:
        processed: 特征工程后的 DataFrame
        drop_small_open: 是否过滤开盘价过小的样本
        use_quantile_label: True=使用每日组内分位数rank（P1改进），
                            False=使用超额收益（原始方案）
        use_mixed_label: True=混合标签（P1改进），同时保留排序能力和绝对收益感知
        mixed_alpha: 混合标签中 rank_label 的权重（0~1），越大越偏向排序

    分位数方案 (use_quantile_label=True):
      每日组内计算绝对收益的百分位rank(0~1)作为排序标签。
      优势：天然抹平牛熊市方向偏差，任何市场环境只关注"谁比谁好"。
      缺陷：牛市+10%和熊市-10%的标签分布完全一样。

    混合标签 (use_mixed_label=True) — 推荐 🌟:
      综合分位数排序质量和绝对收益信号：
        label = alpha * rank_label + (1-alpha) * abs_label_norm
      - alpha=0.7: 排序质量仍占主导
      - (1-alpha)=0.3: 保留绝对收益的符号和大致幅度
      - abs_label 经 clip(-2,2) 防止极端值污染分布
    """
    processed = processed.copy()
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    # Step 1: 计算绝对收益（保留作为 return_head 回归目标）
    processed['label_abs'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label_abs'])

    if use_mixed_label:
        # ─── P1 改进：混合标签（排序 + 绝对收益）───
        # alpha * rank_label 保留排序质量
        rank_label = processed.groupby('日期')['label_abs'].rank(pct=True)
        # (1-alpha) * abs_label 保留收益方向和幅度
        abs_mean = processed['label_abs'].abs().mean()
        abs_label = processed['label_abs'] / (abs_mean + 1e-8)
        abs_label = abs_label.clip(-2, 2)  # 防止极端值
        processed['label'] = mixed_alpha * rank_label + (1 - mixed_alpha) * abs_label
    elif use_quantile_label:
        # ─── 纯分位数排序标签 ───
        processed['label'] = processed.groupby('日期')['label_abs'].rank(pct=True)
    else:
        # ─── 原始方案：超额收益 ───
        day_means = processed.groupby('日期')['label_abs'].transform('mean')
        processed['label'] = processed['label_abs'] - day_means

    # Step 3: 构建辅助任务标签（始终基于绝对收益）
    processed['direction'] = (processed['label_abs'] > 0).astype(np.float32)
    processed['volatility'] = np.abs(processed['label_abs']).astype(np.float32)

    # ─── ★ 市场方向标签（用于市场聚合架构）─────
    # 每日计算：所有股票 label_abs 的均值 > 0 → 市场上涨
    daily_market = processed.groupby('日期')['label_abs'].mean()
    processed['market_label'] = processed['日期'].map(
        lambda d: 1.0 if daily_market.get(d, 0) > 0 else 0.0
    ).astype(np.float32)

    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


# ══════════════════════════════════════════════════════════
# 数据增强 (阶段2: 对抗训练)
# ══════════════════════════════════════════════════════════

def augment_batch(sequences, targets, masks=None, config=None,
                  is_training=True):
    """
    对 batch 数据进行增强。

    增强策略:
    1. 时序掩码: 随机连续 mask 10-20% 时间步
    2. 特征噪声: 对数值特征添加高斯噪声
    3. 股票丢弃: 随机丢弃部分股票 (模拟成分股调整)

    Args:
        sequences: [B, N, L, F] 特征序列
        targets: [B, N] 目标标签
        masks: [B, N] 掩码
        config: 配置字典
        is_training: 是否训练模式

    Returns:
        augmented_sequences, targets, (optional) masks
    """
    if not is_training or config is None:
        return sequences, targets, masks

    prob = config.get('augment_prob', 0.5)
    if random.random() > prob:
        return sequences, targets, masks

    B, N, L, F = sequences.shape
    device = sequences.device
    aug_seq = sequences.clone()

    # 1. 时序掩码: 随机 mask 连续时间步
    time_mask_ratio = config.get('time_mask_ratio', 0.15)
    if time_mask_ratio > 0:
        for b in range(B):
            for n in range(N):
                # 跳过 padding 股票
                if masks is not None and masks[b, n] == 0:
                    continue
                mask_len = int(L * time_mask_ratio * (0.5 + random.random()))
                if mask_len > 0:
                    start = random.randint(0, L - mask_len)
                    aug_seq[b, n, start:start+mask_len, :] = 0

    # 2. 特征噪声: N(0, std*5e-4~1e-2)
    noise_std = config.get('feature_noise_std', 0.005)
    if noise_std > 0:
        noise = torch.randn_like(aug_seq) * noise_std * (0.1 + random.random())
        # 不对 instrument 列 (第0列) 加噪声
        noise[:, :, :, 0] = 0  # instrument 是整数索引
        aug_seq = aug_seq + noise

    # 3. 股票丢弃: 随机 mask 部分股票
    stock_dropout = config.get('stock_dropout_ratio', 0.2)
    if stock_dropout > 0 and masks is not None:
        for b in range(B):
            valid_stocks = masks[b].nonzero(as_tuple=True)[0]
            n_drop = max(1, int(len(valid_stocks) * stock_dropout * random.random()))
            if n_drop > 0 and len(valid_stocks) > n_drop + 10:
                drop_indices = valid_stocks[torch.randperm(len(valid_stocks))[:n_drop]]
                masks = masks.clone() if masks is not None else None
                if masks is not None:
                    masks[b, drop_indices] = 0

    return aug_seq, targets, masks


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    # P1 改进：支持分位数标签 + 混合标签
    use_quantile = config.get('use_quantile_label', False)
    use_mixed = config.get('use_mixed_label', False)
    mixed_alpha = config.get('mixed_label_alpha', 0.7)
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open,
                                       use_quantile_label=use_quantile,
                                       use_mixed_label=use_mixed,
                                       mixed_alpha=mixed_alpha)
    if use_mixed:
        print(f"  标签模式: 混合标签 (alpha={mixed_alpha}, rank + abs)")
    elif use_quantile:
        print(f"  标签模式: 分位数 rank (quantile label)")
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# 加权的排序损失函数（融合 LambdaRank 思想 + Top-K Precision）
class WeightedRankingLoss(nn.Module):
    """
    组合排序损失 — 融合 Listwise + Pairwise + NDCG + Precision@K + Portfolio.

    子损失:
        - Listwise: Softmax 交叉熵 (全局排序)
        - Pairwise: 成对比较损失 (LambdaRank ΔNDCG 权重)
        - NDCG@K: 可微近似 NDCG (use_gumbel 控制 Gumbel-Softmax)
        - Precision@K: Soft Top-K 命中率
        - Portfolio Return: Gumbel-Softmax 松弛选股, 直接最大化选中股票收益期望

    Parameters
    ----------
    temperature : float
        Softmax 温度, 越低→越接近 hard ranking.
    k : int
        Top-K 参数 (默认 5).
    weight_factor : float
        LambdaRank ΔNDCG 权重缩放因子.
    pairwise_weight : float
        成对损失权重.
    ndcg_weight : float
        NDCG 近似损失权重.
    precision_weight : float
        Precision@K 损失权重.
    portfolio_weight : float
        Portfolio Return Loss 权重 (0=禁用).
    portfolio_temperature : float
        Gumbel-Softmax 温度 (用于 Portfolio Loss).
    """
    def __init__(self, temperature=1.0, k=5, weight_factor=3.0,
                 pairwise_weight=1, base_weight=1.0, ndcg_weight=0.3,
                 precision_weight=0.5, use_exact_lambda=True,
                 use_gumbel=False, portfolio_weight=0.15,
                 portfolio_temperature=0.5):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight
        self.ndcg_weight = ndcg_weight
        self.precision_weight = precision_weight
        self.use_exact_lambda = use_exact_lambda
        self.portfolio_weight = portfolio_weight
        self.portfolio_temperature = portfolio_temperature
        self.ndcg_loss_fn = NDCGApproxLoss(k=k, temperature=temperature,
                                           use_gumbel=use_gumbel)

    def _compute_exact_lambda_weights(self, y_true, k):
        """使用精确 ΔNDCG 计算 Lambda 权重"""
        batch_size, num_items = y_true.shape
        device = y_true.device
        weights = torch.full_like(y_true, fill_value=self.base_weight)

        for i in range(batch_size):
            # 计算 ideal DCG
            true_sorted, true_indices = torch.sort(y_true[i], descending=True)
            positions = torch.arange(1, k + 1, device=device, dtype=torch.float32)
            discounts = 1.0 / torch.log2(positions + 1.0)
            ideal_dcg = (true_sorted[:k] * discounts).sum()

            if ideal_dcg < 1e-8:
                continue

            # 为每个 top-k 位置的股票计算 |ΔNDCG|
            for rank, idx in enumerate(true_indices[:k]):
                # 该股票从位置 rank 移到位置 0 的 ΔDCG
                pos_rank = float(rank + 1)
                dcg_change = true_sorted[rank] * (1.0 / math.log2(2.0) - 1.0 / math.log2(pos_rank + 1.0))
                ndcg_change = abs(dcg_change) / (ideal_dcg + 1e-8)
                # 缩放并加权
                lambda_w = self.weight_factor * ndcg_change * k
                weights[i, idx] = self.base_weight + lambda_w

        return weights

    def _compute_approx_precision(self, y_pred, y_true, k):
        """Soft Precision@K 损失"""
        batch_size, num_items = y_pred.shape
        # 真实 top-k (hard)
        _, true_topk = torch.topk(y_true, k, dim=1)
        # 预测的 soft 概率
        pred_prob = F.softmax(y_pred / self.temperature, dim=1)

        total_precision = 0.0
        for i in range(batch_size):
            # 预测在 true top-k 中的概率质量
            precision_i = pred_prob[i, true_topk[i]].sum()
            total_precision += precision_i

        avg_precision = total_precision / batch_size
        # 返回值越小越好
        return 1.0 - avg_precision

    def _portfolio_return_loss(self, y_pred, y_true_abs, masks, k=None):
        """
        Portfolio-Level 损失 (P1改进): 用 Soft Top-K 近似选择，
        直接最大化选中股票的真实绝对收益期望。

        解决纯排序损失的排列等变问题：
        - 排序损失给所有分数加同一常数，损失不变
        - Portfolio loss 让模型直接关心"选出的股票能赚多少"

        Args:
            y_pred: [B, N] 预测分数
            y_true_abs: [B, N] 真实绝对收益（label_abs）
            masks: [B, N] 有效股票掩码
            k: Top-K 数量
        """
        if k is None:
            k = self.k

        batch_size, num_items = y_pred.shape
        device = y_pred.device
        total_loss = 0.0
        n_valid = 0

        for i in range(batch_size):
            if masks is not None:
                valid_mask = masks[i] > 0.5
                n_valid_items = valid_mask.sum().item()
                if n_valid_items < k:
                    continue
                pred_i = y_pred[i][valid_mask]
                true_i = y_true_abs[i][valid_mask]
            else:
                pred_i = y_pred[i]
                true_i = y_true_abs[i]
                n_valid_items = pred_i.shape[0]

            if n_valid_items < k:
                continue

            # Gumbel-Softmax 将离散 Top-K 松弛为连续概率
            pred_prob = F.gumbel_softmax(
                pred_i.unsqueeze(0),
                tau=self.portfolio_temperature,
                hard=False, dim=1
            ).squeeze(0)

            _, topk_indices = torch.topk(pred_prob, k, dim=0)
            selected_returns = true_i[topk_indices]

            mean_ret = selected_returns.mean()
            # 负收益额外惩罚
            penalty = F.relu(-mean_ret) * 2.0
            loss_i = -mean_ret + penalty

            total_loss += loss_i
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / n_valid

    def listwise_loss(self, y_pred, y_true, weights):
        """加权的Listwise损失"""
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)

        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        ce_loss = (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()

        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        """加权的Pairwise损失"""
        batch_size, num_items = y_pred.size()

        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)

        mask = (true_diff != 0).float()
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)

        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        weighted_loss = pairwise_loss * mask * weight_matrix

        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()

        return loss

    def forward(self, y_pred, y_true, y_true_abs=None, masks=None):
        """
        y_pred: [batch, num_items] 预测分数
        y_true: [batch, num_items] 排序标签（超额收益或分位数rank）
        y_true_abs: [batch, num_items] 绝对收益（用于portfolio loss, 可选）
        masks: [batch, num_items] 有效股票掩码（可选）
        """
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)
        device = y_true.device

        # 1. 识别 top-k 样本
        _, top_indices = torch.topk(y_true, k, dim=1)

        # 2. 权重计算: 精确 LambdaRank 或指数衰减
        if self.use_exact_lambda:
            weights = self._compute_exact_lambda_weights(y_true, k)
        else:
            weights = torch.full_like(y_true, fill_value=self.base_weight)
            for i in range(batch_size):
                for rank, idx in enumerate(top_indices[i]):
                    lambda_weight = self.weight_factor * math.exp(-rank / k)
                    weights[i, idx] = self.base_weight + lambda_weight

        # 3. 计算各类损失
        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        ndcg = self.ndcg_loss_fn(y_pred, y_true)
        precision = self._compute_approx_precision(y_pred, y_true, k)

        # 4. Portfolio Return 损失（P1改进：直接优化绝对收益期望）
        portfolio_ret = 0.0
        if self.portfolio_weight > 0 and y_true_abs is not None:
            portfolio_ret = self._portfolio_return_loss(
                y_pred, y_true_abs, masks, k
            )
            if isinstance(portfolio_ret, torch.Tensor):
                portfolio_ret = portfolio_ret.item()

        # 5. 组合
        total_loss = (listwise +
                      self.pairwise_weight * pairwise +
                      self.ndcg_weight * ndcg +
                      self.precision_weight * precision +
                      self.portfolio_weight * (portfolio_ret if isinstance(portfolio_ret, (int, float)) else 0.0))

        return total_loss, {
            'listwise': listwise.item(),
            'pairwise': pairwise.item(),
            'ndcg': ndcg.item(),
            'precision': precision.item(),
            'portfolio': float(portfolio_ret) if isinstance(portfolio_ret, (int, float)) else 0.0,
        }

def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算新的评估指标：Top 5 收益之和，以及与理论最高值和随机值的比值"""
    batch_size = y_pred.size(0)
    
    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        
        if valid_indices.numel() < k:
            continue
            
        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices] # This is the 5-day return
        
        # 1. Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        
        # 2. True Top 5 (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        
        # 3. Random 5 (Expected Value)
        # Expected sum = 5 * mean(all valid returns)
        random_return_sum = k * valid_true.mean().item()
        
        # 计算每个样本的比例与稳定化 final_score
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        
    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }
    
    # 比值用逐样本均值，降低极端日影响
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0
    
    return metrics

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类（含辅助标签）"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices, aux_labels=None):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
        self.aux_labels = aux_labels  # list of dicts: {'direction': [...], 'volatility': [...]}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            'sequences': torch.FloatTensor(self.sequences[idx]),
            'targets': torch.FloatTensor(self.targets[idx]),
            'relevance': torch.LongTensor(self.relevance_scores[idx]),
            'stock_indices': torch.LongTensor(self.stock_indices[idx])
        }
        if self.aux_labels is not None:
            item['direction'] = torch.FloatTensor(self.aux_labels[idx]['direction'])
            item['volatility'] = torch.FloatTensor(self.aux_labels[idx]['volatility'])
            if 'return_abs' in self.aux_labels[idx]:
                item['return_abs'] = torch.FloatTensor(self.aux_labels[idx]['return_abs'])
            if 'market_label' in self.aux_labels[idx]:
                item['market_label'] = torch.FloatTensor(self.aux_labels[idx]['market_label'])
        return item

def collate_fn(batch):
    """
    自定义 collate 函数 — 处理变长股票数量的 batch.

    各股票日的有效股票数可能不同 (部分股票停牌/退市).
    此函数将不等长序列 padding 到 batch 内最大股票数.

    Returns
    -------
    dict
        'sequences': [B, max_N, seq_len, F] padded float tensor
        'targets': [B, max_N] padded float tensor
        'relevance': [B, max_N] padded float tensor
        'masks': [B, max_N] 二进制掩码 (1=有效, 0=padding)
        'direction'/'volatility'/'return_abs'/'market_label' (可选)
    """
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    has_aux = 'direction' in batch[0]
    has_return_abs = has_aux and 'return_abs' in batch[0]
    has_market_label = has_aux and 'market_label' in batch[0]

    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)

    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    padded_direction = [] if has_aux else None
    padded_volatility = [] if has_aux else None
    padded_return_abs = [] if has_return_abs else None
    padded_market_label = [] if has_market_label else None
    masks = []

    for i, (seq, tgt, rel, stock_idx) in enumerate(zip(sequences, targets, relevance, stock_indices)):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)

        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)

            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)

            if has_aux:
                dir_pad = torch.zeros(pad_size)
                vol_pad = torch.zeros(pad_size)
                batch[i]['direction'] = torch.cat([batch[i]['direction'], dir_pad], dim=0)
                batch[i]['volatility'] = torch.cat([batch[i]['volatility'], vol_pad], dim=0)
            if has_return_abs:
                ret_pad = torch.zeros(pad_size)
                batch[i]['return_abs'] = torch.cat([batch[i]['return_abs'], ret_pad], dim=0)
            if has_market_label:
                mkt_pad = torch.zeros(pad_size)
                batch[i]['market_label'] = torch.cat([batch[i]['market_label'], mkt_pad], dim=0)

        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
        if has_aux:
            padded_direction.append(batch[i]['direction'])
            padded_volatility.append(batch[i]['volatility'])
        if has_return_abs:
            padded_return_abs.append(batch[i]['return_abs'])
        if has_market_label:
            padded_market_label.append(batch[i]['market_label'])

    result = {
        'sequences': torch.stack(padded_sequences),
        'targets': torch.stack(padded_targets),
        'relevance': torch.stack(padded_relevance),
        'stock_indices': torch.stack(padded_stock_indices),
        'masks': torch.stack(masks)
    }
    if has_aux:
        result['direction'] = torch.stack(padded_direction)
        result['volatility'] = torch.stack(padded_volatility)
    if has_return_abs:
        result['return_abs'] = torch.stack(padded_return_abs)
    if has_market_label:
        result['market_label'] = torch.stack(padded_market_label)
    return result

# 排序训练函数（多任务版本）
def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer,
                        aux_dir_criterion=None, aux_vol_criterion=None,
                        aux_dir_weight=0.1, aux_vol_weight=0.1,
                        aux_return_criterion=None, aux_return_weight=0.0, config=None):
    """
    单 epoch 训练循环 — 多任务排序学习.

    执行流程:
        1. 数据增强 (时序掩码/特征噪声/股票丢弃)
        2. 前向传播 (model(sequences, return_aux=True))
        3. 损失计算:
           - 主损失: WeightedRankingLoss (排序)
           - 辅助: BCE(方向) + Huber(波动) + Huber(绝对收益)
           - 市场: BCE(涨/跌预测, market_logits vs market_label)
           - Portfolio: 组合收益损失
        4. 梯度累积 → 反向传播

    Returns
    -------
    avg_loss : float
        平均 batch 损失.
    metrics : dict
        训练指标 (含各子损失分量).
    """
    model.train()
    total_loss = 0
    total_metrics = {}
    total_loss_components = {}
    local_step = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        relevance = batch['relevance'].to(device)
        masks = batch['masks'].to(device)

        # ─── 数据增强 (阶段2) ──────────────────────
        sequences, targets, masks = augment_batch(
            sequences, targets, masks, config, is_training=True
        )

        optimizer.zero_grad()

        # 模型预测（训练时返回辅助任务输出，传入 stock_masks 用于市场聚合）
        outputs, aux = model(sequences, return_aux=True, stock_masks=masks)  # outputs: [B, M], aux: dict

        # 应用mask
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks

        # 计算排序损失
        batch_loss = None
        batch_loss_components = {}
        batch_size = sequences.size(0)

        # 获取绝对收益（用于portfolio loss）
        has_return_abs_batch = 'return_abs' in batch
        if has_return_abs_batch:
            batch_return_abs = batch['return_abs'].to(device)  # [B, N]
            masked_return_abs = batch_return_abs * masks
        else:
            masked_return_abs = None

        for i in range(batch_size):
            mask_i = masks[i]
            valid_indices = mask_i.nonzero().squeeze()

            if valid_indices.numel() == 0:
                continue

            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)

            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]

            # 准备 portfolio loss 的参数
            valid_return_abs = None
            valid_masks = None
            if has_return_abs_batch and masked_return_abs is not None:
                valid_return_abs = masked_return_abs[i][valid_indices].unsqueeze(0)
                valid_masks = torch.ones(1, len(valid_indices), device=device)

            if len(valid_pred) > 1:
                loss, loss_comps = criterion(
                    valid_pred.unsqueeze(0),
                    valid_relevance.unsqueeze(0),
                    y_true_abs=valid_return_abs,
                    masks=valid_masks
                )
                batch_loss = batch_loss + loss if batch_loss is not None else loss
                for k, v in loss_comps.items():
                    batch_loss_components[k] = batch_loss_components.get(k, 0) + v

        if batch_loss is not None:
            batch_loss = batch_loss / batch_size

            # 辅助任务损失
            aux_loss_total = 0.0
            if aux_dir_criterion is not None and 'direction' in batch:
                dir_labels = batch['direction'].to(device)
                # 展平 batch*num_stocks 维度
                aux_dir = aux['direction']  # [B*N]
                # 只对有效位置计算
                flat_masks = masks.view(-1)
                valid_aux_dir = aux_dir[flat_masks > 0]
                valid_dir_labels = dir_labels.view(-1)[flat_masks > 0]
                if len(valid_aux_dir) > 0:
                    aux_dir_loss = aux_dir_criterion(valid_aux_dir, valid_dir_labels)
                    batch_loss = batch_loss + aux_dir_weight * aux_dir_loss
                    aux_loss_total += aux_dir_weight * aux_dir_loss.item()

            if aux_vol_criterion is not None and 'volatility' in batch:
                vol_labels = batch['volatility'].to(device)
                aux_vol = aux['volatility']  # [B*N]
                flat_masks = masks.view(-1)
                valid_aux_vol = aux_vol[flat_masks > 0]
                valid_vol_labels = vol_labels.view(-1)[flat_masks > 0]
                if len(valid_aux_vol) > 0:
                    aux_vol_loss = aux_vol_criterion(valid_aux_vol, valid_vol_labels)
                    batch_loss = batch_loss + aux_vol_weight * aux_vol_loss
                    aux_loss_total += aux_vol_weight * aux_vol_loss.item()

            # 绝对收益回归损失（用于收益门控）
            if aux_return_criterion is not None and aux_return_weight > 0 and 'return_abs' in batch:
                return_labels = batch['return_abs'].to(device)
                aux_return_pred = aux['return_abs']  # [B*N]
                flat_masks = masks.view(-1)
                valid_aux_return = aux_return_pred[flat_masks > 0]
                valid_return_labels = return_labels.view(-1)[flat_masks > 0]
                if len(valid_aux_return) > 0:
                    aux_return_loss = aux_return_criterion(valid_aux_return, valid_return_labels)
                    batch_loss = batch_loss + aux_return_weight * aux_return_loss
                    aux_loss_total += aux_return_weight * aux_return_loss.item()

            # ─── ★ 市场方向损失（市场聚合架构）──────
            market_loss_weight = config.get('market_loss_weight', 0.2)
            if market_loss_weight > 0 and 'market_logits' in aux and 'market_label' in batch:
                market_labels = batch['market_label'].to(device)  # [B, N]
                market_logits = aux['market_logits']  # [B]
                # 每个 batch 只有一个市场方向标签（所有股票相同），取第一个有效股票的值
                # 用 masks 找到每行的第一个有效股票
                batch_size_actual = market_logits.shape[0]
                market_labels_per_batch = []
                for b in range(batch_size_actual):
                    valid_idx = (masks[b] > 0.5).nonzero(as_tuple=True)[0]
                    if len(valid_idx) > 0:
                        market_labels_per_batch.append(market_labels[b, valid_idx[0]])
                    else:
                        market_labels_per_batch.append(torch.tensor(0.0, device=device))
                market_labels_vec = torch.stack(market_labels_per_batch)  # [B]
                market_loss = F.binary_cross_entropy_with_logits(
                    market_logits, market_labels_vec
                )
                batch_loss = batch_loss + market_loss_weight * market_loss
                aux_loss_total += market_loss_weight * market_loss.item()

            batch_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            if writer:
                writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
            optimizer.step()

            total_loss += batch_loss.item()
            for k, v in batch_loss_components.items():
                total_loss_components[k] = total_loss_components.get(k, 0) + v

            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v

            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
                if aux_loss_total > 0:
                    writer.add_scalar('train/aux_loss', aux_loss_total, global_step=epoch*len(dataloader)+local_step)

    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_loss, total_metrics

def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            
            # 模型预测
            outputs = model(sequences)
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                
                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(len(valid_true), 0, -1, device=device, dtype=torch.float32)
                    relevance_scores = relevance_scores.detach()

                    # criterion 现在返回 (loss, loss_components)
                    loss, _ = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5, config=None):
    """
    预测某一天最优 Top-K 股票的排序分数。

    改进：使用 softmax 权重 + 波动率惩罚替代等权重。
    借鉴「7355608」的分数融合和「O_O」的风险调整思想。
    """
    model.eval()

    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()

    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []

    for stock_code in data['股票代码'].unique():
        stock_history = data[
            (data['股票代码'] == stock_code) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])

    if len(day_sequences) == 0:
        return []

    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(sequences)  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()

    # ─── 新后处理：分数校准 + 波动率惩罚 ──────────
    candidate_k = config.get('post_top_k', 10) if config else 10
    use_vol_penalty = config.get('use_volatility_penalty', True) if config else True
    temperature = 2.0

    # 计算每只股票的波动率（用于惩罚项）
    volatilities = None
    if use_vol_penalty:
        # 用最近 20 天涨跌幅的标准差作为波动率估计
        vols = []
        for i, seq in enumerate(day_sequences):
            # 计算 returns: close_t / close_{t-1} - 1
            close_prices = seq[:, 2]  # features[2] = '收盘' (features[0]='instrument', features[1]='开盘')
            returns = np.diff(close_prices) / (close_prices[:-1] + 1e-12)
            vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            vols.append(vol)
        volatilities = np.array(vols)

    # 使用 optimize_weights 进行后处理
    top_indices, top_weights = optimize_weights(
        scores, volatilities=volatilities,
        top_k=top_k, candidate_k=candidate_k,
        use_volatility_penalty=use_vol_penalty,
        temperature=temperature
    )

    top_stocks = []
    for rank, (idx, weight) in enumerate(zip(top_indices, top_weights)):
        top_stocks.append({
            'stock_code': day_stock_codes[idx],
            'predicted_score': float(scores[idx]),
            'weight': float(weight),
            'rank': rank + 1
        })

    return top_stocks

def save_predictions(top_stocks, output_path, use_weighted=True):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score'],
            '权重': stock.get('weight', 0.2)
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")

    # 同时保存竞赛标准格式（stock_id, weight）
    if use_weighted:
        comp_format = []
        for stock in top_stocks:
            comp_format.append({
                'stock_id': stock['stock_code'],
                'weight': stock.get('weight', 0.2)
            })
        comp_df = pd.DataFrame(comp_format)
        comp_output = output_path.replace('.csv', '_comp.csv')
        comp_df.to_csv(comp_output, index=False, encoding='utf-8')
        print(f"竞赛格式已保存到: {comp_output}")


def split_train_val_by_last_month(df, sequence_length, val_months=None):
    """按最后 N 个月做验证集划分，并为验证集补充序列上下文。"""
    if val_months is None:
        val_months = config.get('val_months', 3)

    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=val_months)).normalize()

    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后{val_months}个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")
    n_train_days = train_df['日期'].nunique()
    n_val_days = val_df['日期'].nunique()
    print(f"训练集天数: {n_train_days}, 验证集天数: {n_val_days}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start

# 主程序
def main(seed=None):
    set_seed(config.get('seed', 42))
    seed = seed if seed is not None else config.get('seed', 42)
    config['seed'] = seed
    set_seed(seed)

    # 动态设置输出目录（含种子）
    base_output_dir = config['output_dir']
    if seed != 42:
        config['output_dir'] = f"{base_output_dir}_seed{seed}"
    output_dir = config['output_dir']
    os.makedirs(output_dir,exist_ok=True)
    # 保存在output_dir中保存当前的配置文件，以便复现
    data_path = config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"设备: {device} | 种子: {seed} | 输出: {output_dir}")

    # 0. 加载基本面数据（如果启用）
    if config.get('use_fundamentals', False):
        print("加载基本面数据...")
        load_fundamentals()  # 预加载到缓存

    # 1. 数据加载
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'], config.get('val_months', 3))
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    
    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)
    
    # 3. 标准化
    scaler = StandardScaler()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    # 丢弃nan数据
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    # 然后再缩放
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    
    # 4. 创建排序数据集（含辅助标签）
    train_sequences, train_targets, train_relevance, train_stock_indices, train_aux = create_ranking_dataset_vectorized(
        train_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('train_ranking_data_path')
    )
    val_sequences, val_targets, val_relevance, val_stock_indices, val_aux = create_ranking_dataset_vectorized(
        val_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('val_ranking_data_path'),
        min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")

    # 5. 创建排序数据集和数据加载器
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices, train_aux)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices, val_aux)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0),
        ndcg_weight=config.get('ndcg_weight', 0.3),
        precision_weight=config.get('precision_weight', 0.5),
        use_exact_lambda=config.get('use_exact_lambda', True),
        use_gumbel=config.get('use_gumbel_ndcg', False),
        portfolio_weight=config.get('portfolio_loss_weight', 0.15),
        portfolio_temperature=config.get('portfolio_temperature', 0.5),
    )
    # 辅助任务损失
    aux_dir_criterion = nn.BCEWithLogitsLoss()
    aux_vol_criterion = nn.MSELoss()
    aux_return_criterion = nn.SmoothL1Loss(beta=0.1)  # Huber loss for absolute return regression (robust to outliers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

    # Warmup + Linear Decay 学习率调度
    warmup_epochs = config.get('warmup_epochs', 5)
    warmup_start_lr = config.get('warmup_start_lr', 1e-6)
    # 使用 LambdaLR 实现 warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return warmup_start_lr / config['learning_rate'] + \
                   (1.0 - warmup_start_lr / config['learning_rate']) * epoch / warmup_epochs
        else:
            # Linear decay from 1.0 to 0.2 over remaining epochs
            remaining = config['num_epochs'] - warmup_epochs
            progress = (epoch - warmup_epochs) / max(remaining, 1)
            return 1.0 - progress * 0.8  # end_factor=0.2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 8. 排序模型训练（多任务 + early stopping）
    if is_train:
        best_score = -float('inf')
        best_epoch = -1
        patience_counter = 0
        early_stopping_patience = config.get('early_stopping_patience', 12)

        aux_dir_weight = config.get('aux_direction_weight', 0.1)
        aux_vol_weight = config.get('aux_volatility_weight', 0.1)
        aux_return_weight = config.get('aux_return_weight', 0.3)

        for epoch in range(config['num_epochs']):
            current_lr = scheduler.get_last_lr()[0]
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} (LR: {current_lr:.2e}) ===")

            # 训练（多任务）
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer,
                aux_dir_criterion=aux_dir_criterion,
                aux_vol_criterion=aux_vol_criterion,
                aux_dir_weight=aux_dir_weight,
                aux_vol_weight=aux_vol_weight,
                aux_return_criterion=aux_return_criterion,
                aux_return_weight=aux_return_weight,
                config=config
            )

            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")

            # 验证
            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, epoch
            )

            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")

            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', current_lr, global_step=epoch)

            # 保存最佳模型（基于 final score）+ early stopping
            current_final_score = eval_metrics.get('final_score', 0.0)
            if current_final_score > best_score:
                best_score = current_final_score
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"✅ 保存最佳模型 - final score: {best_score:.4f}")
            elif early_stopping_patience > 0:
                patience_counter += 1
                print(f"  早停计数: {patience_counter}/{early_stopping_patience} (best: {best_score:.4f} @ epoch {best_epoch})")
                if patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final score: {best_score:.4f}")
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\\nBest final_score: {best_score:.6f}\\n")

        if writer:
            writer.close()

        return best_score

def train_ensemble():
    """阶段4: 多模型集成训练 — 使用不同随机种子训练多个模型"""
    seeds = config.get('ensemble_seeds', [42, 123, 456])
    results = {}
    for s in seeds:
        print(f"\n{'='*60}")
        print(f"Training ensemble model with seed={s}")
        print(f"{'='*60}")
        score = main(seed=s)
        results[str(s)] = float(score)

    # 保存集成结果
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ensemble_results.json'), 'w') as f:
        json.dump({'results': results, 'mean_score': float(np.mean(list(results.values())))}, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Ensemble training complete!")
    print(f"Individual scores: {results}")
    print(f"Mean score: {np.mean(list(results.values())):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # 支持命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--ensemble':
            train_ensemble()
        elif sys.argv[1] == '--seed' and len(sys.argv) > 2:
            seed = int(sys.argv[2])
            best_score = main(seed=seed)
            print(f"\n########## Training complete! Best final score: {best_score:.4f} ##########")
        else:
            print(f"Usage: python train.py  (single seed={config.get('seed',42)})")
            print(f"       python train.py --seed N  (single seed)")
            print(f"       python train.py --ensemble  (multi-seed ensemble)")
    else:
        best_score = main()
        print(f"\n########## Training complete! Best final score: {best_score:.4f} ##########")
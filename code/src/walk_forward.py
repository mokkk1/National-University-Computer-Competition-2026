"""
walk_forward.py — Walk-Forward 滚动窗口训练框架

核心思想：
  将数据按时间顺序划分为多个 (训练窗口, 验证窗口) 对，
  每个窗口独立训练模型，最终用所有窗口模型做集成预测。

目的：
  - 阻止在单一固定验证窗口上的元过拟合
  - 评估模型在不同市场环境下的泛化能力
  - 选择在多个窗口中稳定表现的超参数配置

用法：
  python code/src/walk_forward.py                           # 使用默认配置训练
  python code/src/walk_forward.py --config light            # 使用轻量配置
  python code/src/walk_forward.py --config standard         # 使用标准V7配置
  python code/src/walk_forward.py --seeds 42,123,456        # 多seed集成
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
import json
import argparse
import multiprocessing as mp
import random
from datetime import datetime, timedelta
from tqdm import tqdm
from tensorboardX import SummaryWriter

# 导入现有模块
from config import config as default_config
from config import LIGHTWEIGHT_CONFIG
from model import StockTransformer, LightweightStockRanker
from utils import (engineer_features_158plus39, create_ranking_dataset_vectorized,
                   build_lazy_ranking_dataset,
                   optimize_weights, select_top_stocks_with_gate)
from train import (feature_cloums_map, feature_engineer_func_map, RankingDataset,
                   collate_fn, WeightedRankingLoss, calculate_ranking_metrics,
                   augment_batch, set_seed, train_ranking_model, evaluate_ranking_model)
from fundamental import (load_fundamentals, engineer_features_all,
                         FUNDAMENTAL_FEATURE_COLS, MOMENTUM_FEATURE_COLS)

# ─── Walk-Forward 窗口定义 ──────────────────────────────
# 每个窗口: (train_end, val_end)，train从数据起始日开始，val为train_end后的N个月
# 窗口逐步向前滚动，模拟真实的"训练→预测未来"场景

def generate_walk_forward_windows(data_start, data_end, val_months=2, step_months=2, min_train_months=6):
    """
    自动生成 Walk-Forward 窗口。

    Args:
        data_start: 数据起始日期 (pd.Timestamp)
        data_end: 数据结束日期 (pd.Timestamp)
        val_months: 每个验证窗口的月数
        step_months: 窗口滚动步长（月）
        min_train_months: 最少训练数据月数

    Returns:
        list of (train_end, val_end) tuples，每个窗口训练集为 [data_start, train_end]，验证集为 [train_end+1day, val_end]
    """
    windows = []
    current_val_end = data_end

    # 从后往前生成窗口
    min_train_date = data_start + pd.DateOffset(months=min_train_months)

    while True:
        val_end = current_val_end
        val_start = val_end - pd.DateOffset(months=val_months)
        train_end = val_start - pd.tseries.offsets.BDay(1)

        if train_end < min_train_date:
            break

        windows.append((train_end.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d')))
        current_val_end = current_val_end - pd.DateOffset(months=step_months)

    # 反转使得窗口按时间顺序排列（最早→最晚）
    windows.reverse()
    return windows


# ─── 轻量模型配置（用于 Walk-Forward）───────────────────
LIGHT_CONFIG = {
    'sequence_length': 60,
    'feature_num': '158+39',
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'batch_size': 4,
    'num_epochs': 40,
    'learning_rate': 1e-5,
    'dropout': 0.3,
    'max_grad_norm': 5.0,
    # 损失函数
    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 3.0,
    'ndcg_weight': 0.3,
    'precision_weight': 0.5,
    'use_exact_lambda': True,
    'use_gumbel_ndcg': False,
    # TCN — 关闭以减少过拟合
    'use_tcn': False,
    # 特征交互 — 关闭
    'use_feature_interaction': False,
    # 辅助任务 — 加强作为正则化
    'aux_direction_weight': 0.3,
    'aux_volatility_weight': 0.3,
    'aux_return_weight': 0.3,
    # 基本面/动量 — 只用基础特征
    'use_fundamentals': False,
    'use_momentum_features': False,
    # 数据增强 — 加强
    'augment_prob': 0.6,
    'time_mask_ratio': 0.15,
    'feature_noise_std': 0.005,
    'stock_dropout_ratio': 0.2,
    # 训练策略
    'warmup_epochs': 3,
    'warmup_start_lr': 1e-6,
    'early_stopping_patience': 10,
    'val_months': 2,
    # P1 改进：分位数标签
    'use_quantile_label': True,   # 使用分位数 rank 标签
    # 后处理
    'post_top_k': 10,
    'use_volatility_penalty': True,
    'use_return_gate': False,  # 轻量配置不使用收益门控
    'seed': 42,
    'output_dir': None,  # 运行时设置
    'data_path': None,
}

STANDARD_CONFIG = {
    'sequence_length': 60,
    'feature_num': '158+39+fundamental+momentum',
    'd_model': 256,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 512,
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 1e-5,
    'dropout': 0.15,
    'max_grad_norm': 5.0,
    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 3.0,
    'ndcg_weight': 0.3,
    'precision_weight': 0.5,
    'use_exact_lambda': True,
    'use_gumbel_ndcg': False,
    'use_tcn': True,
    'tcn_kernel_sizes': [3, 5, 7],
    'tcn_dropout': 0.1,
    'use_feature_interaction': True,
    'fi_rank': 64,
    'aux_direction_weight': 0.1,
    'aux_volatility_weight': 0.1,
    'aux_return_weight': 0.3,
    'use_fundamentals': True,
    'use_momentum_features': True,
    'momentum_features': ['ret5', 'ret20', 'vol20', 'sharpe5'],
    'augment_prob': 0.4,
    'time_mask_ratio': 0.1,
    'feature_noise_std': 0.003,
    'stock_dropout_ratio': 0.15,
    'warmup_epochs': 5,
    'warmup_start_lr': 1e-6,
    'early_stopping_patience': 12,
    'val_months': 2,
    # P1 改进：分位数标签
    'use_quantile_label': True,
    # 宏观特征 + 行业 Embedding
    'use_macro_features': True,
    'use_industry_embedding': True,
    'num_industries': 31,
    'industry_emb_dim': 16,
    'post_top_k': 10,
    'use_volatility_penalty': True,
    'use_return_gate': True,
    'min_return_threshold': 0.0,
    'return_gate_fallback': 'equal',
    'seed': 42,
    'output_dir': None,
    'data_path': None,
}


def get_config_by_name(name: str) -> dict:
    """根据名称获取配置"""
    configs = {
        'light': LIGHT_CONFIG,
        'standard': STANDARD_CONFIG,
        'v7': default_config,
        'v8_improved': default_config,
        'lightweight': LIGHTWEIGHT_CONFIG,  # V9: 轻量架构 + 方向分类
    }
    if name not in configs:
        raise ValueError(f"未知配置: {name}。可选: {list(configs.keys())}")
    cfg = configs[name].copy()
    # 深度拷贝嵌套结构
    return json.loads(json.dumps(cfg))


# ══════════════════════════════════════════════════════════
# 分位数标签构建（P1 改进）
# ══════════════════════════════════════════════════════════

def build_quantile_label(processed, drop_small_open=True):
    """
    使用每日组内分位数 rank 作为排序标签。

    优势：
    - 天然抹平牛熊市的方向偏差
    - 在任何市场环境下都只关注"谁比谁更好"的相对排序
    - 避免选出"跌得少"而非"涨得多"的股票

    同时保留 label_abs 用于 return_head 回归和方向辅助任务。
    """
    processed = processed.copy()
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    # 绝对收益（保留用于辅助任务）
    processed['label_abs'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label_abs'])

    # 每日组内分位数 rank (0~1)，作为排序标签
    processed['label'] = processed.groupby('日期')['label_abs'].rank(pct=True)

    # 辅助任务标签
    processed['direction'] = (processed['label_abs'] > 0).astype(np.float32)
    processed['volatility'] = np.abs(processed['label_abs']).astype(np.float32)

    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


# ══════════════════════════════════════════════════════════
# 日历特征（方向一：季末感知）
# ══════════════════════════════════════════════════════════

def add_calendar_features(df):
    """
    为每行数据添加日历相关特征，帮助模型感知季末效应。

    新增特征:
      - days_to_qe: 距离最近季末日(3/31, 6/30, 9/30, 12/31)的天数，归一化到 [0, 1]
      - is_qe_month: 当前月份是否为季末月 (3/6/9/12)
    """
    df = df.copy()
    dates = pd.to_datetime(df['日期'])

    def _days_to_qe(d):
        m, y = d.month, d.year
        if m <= 3:
            qe = pd.Timestamp(year=y, month=3, day=31)
        elif m <= 6:
            qe = pd.Timestamp(year=y, month=6, day=30)
        elif m <= 9:
            qe = pd.Timestamp(year=y, month=9, day=30)
        else:
            qe = pd.Timestamp(year=y, month=12, day=31)
        return max(0.0, (qe - d).days) / 90.0  # 归一化

    df['days_to_qe'] = dates.apply(_days_to_qe).astype(np.float32)
    df['is_qe_month'] = dates.dt.month.isin([3, 6, 9, 12]).astype(np.float32)

    return df


# ══════════════════════════════════════════════════════════
# Walk-Forward 核心逻辑
# ══════════════════════════════════════════════════════════

def _preprocess_window(df, stockid2idx, config, desc="特征工程", drop_small_open=True):
    """对单个窗口的数据做特征工程和标签构建"""
    feature_num = config.get('feature_num', '158+39')
    feature_engineer = feature_engineer_func_map.get(feature_num, engineer_features_158plus39)
    feature_columns_template = feature_cloums_map.get(feature_num, feature_cloums_map['158+39'])

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"  使用多进程计算特征 ({feature_num})...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError("输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(
            pool.imap(feature_engineer, groups),
            total=len(groups), desc=f"  {desc}"
        ))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['日期'] = pd.to_datetime(processed['日期'])

    # ─── 宏观特征拼接 ────────────────────────────
    use_macro = config.get('use_macro_features', True)
    macro_cols = []
    if use_macro:
        try:
            from macro_industry import load_macro_features, merge_macro_to_stock
            macro_df = load_macro_features()
            if macro_df is not None:
                macro_cols_before = list(processed.columns)
                processed = merge_macro_to_stock(processed, macro_df)
                macro_cols = [c for c in processed.columns if c not in macro_cols_before]
                if macro_cols:
                    print(f"  宏观特征: {len(macro_cols)} 维 ({', '.join(macro_cols[:5])}...)")
        except Exception as e:
            print(f"  宏观特征加载失败: {e}")

    # ─── 行业分类 ────────────────────────────────
    use_industry = config.get('use_industry_embedding', False)
    if use_industry:
        try:
            from macro_industry import add_industry_features
            processed, _ = add_industry_features(processed)
            if 'industry' not in processed.columns:
                processed['industry'] = 0
        except Exception as e:
            print(f"  行业特征添加失败: {e}")
            processed['industry'] = 0

    # ─── 日历特征（★ 方向一：季末感知）─────────────
    use_calendar = config.get('use_calendar_features', False)
    if use_calendar:
        processed = add_calendar_features(processed)
        print(f"  日历特征: days_to_qe, is_qe_month")

    # 股票索引映射
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    # ─── 标签构建（支持分位数/混合标签/市场聚合）───
    use_quantile_label = config.get('use_quantile_label', True)
    use_mixed_label = config.get('use_mixed_label', False)
    use_market_agg = config.get('use_market_aggregation', False)

    # 混合标签或市场聚合架构 → 使用 train.py 的 _build_label_and_clean
    # （内部支持混合标签 + market_label 构建）
    if use_mixed_label or use_market_agg:
        from train import _build_label_and_clean
        processed = _build_label_and_clean(
            processed, drop_small_open=drop_small_open,
            use_quantile_label=use_quantile_label,
            use_mixed_label=use_mixed_label,
            mixed_alpha=config.get('mixed_label_alpha', 0.7)
        )
    elif use_quantile_label:
        processed = build_quantile_label(processed, drop_small_open=drop_small_open)
    else:
        from train import _build_label_and_clean
        processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)

    return processed, feature_columns_template


def train_single_window(train_df, val_df, val_start, config, stockid2idx, window_idx, output_base_dir):
    """
    在单个 Walk-Forward 窗口上训练模型。

    Args:
        train_df: 训练数据
        val_df: 验证数据（含序列上下文）
        val_start: 验证集起始日期
        config: 模型配置
        stockid2idx: 股票代码→索引映射
        window_idx: 窗口编号
        output_base_dir: 输出根目录

    Returns:
        (best_val_score, output_dir) 或 None（训练失败时）
    """
    seed = config.get('seed', 42)
    set_seed(seed)

    window_output_dir = os.path.join(output_base_dir, f'window_{window_idx:02d}')
    os.makedirs(window_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Window {window_idx}: Training model")
    print(f"  Train range: {train_df['日期'].min()} ~ {train_df['日期'].max()}")
    print(f"  Val range:   {val_start} ~ {val_df['日期'].max()}")
    print(f"  Output:      {window_output_dir}")
    print(f"{'='*60}")

    # 特征工程
    print("\n[1/6] 特征工程...")
    train_data, feature_columns = _preprocess_window(
        train_df, stockid2idx, config, desc=f"W{window_idx} 训练特征"
    )
    val_data, _ = _preprocess_window(
        val_df, stockid2idx, config, desc=f"W{window_idx} 验证特征"
    )

    # 获取实际的特征列（包括宏观列和行业列）
    feature_cols = [c for c in feature_columns if c in train_data.columns]
    # 动态识别宏观特征列（日频 + 月频前向填充）
    MACRO_PREFIXES = ('bond', 'shibor', 'north', 'margin', 'usdcny', 'lpr',
                      'cpi', 'ppi', 'pmi', 'm1', 'm2', 'social')
    macro_cols = [c for c in train_data.columns
                  if c.startswith(MACRO_PREFIXES) and c not in feature_cols]
    for mc in macro_cols:
        if mc not in feature_cols:
            feature_cols.append(mc)

    # ★ 日历特征列（方向一）
    calendar_cols = ['days_to_qe', 'is_qe_month']
    for cc in calendar_cols:
        if cc in train_data.columns and cc not in feature_cols:
            feature_cols.append(cc)

    # 行业列单独处理（不标准化）
    has_industry = 'industry' in train_data.columns and config.get('use_industry_embedding', False)
    if has_industry and 'industry' not in feature_cols:
        industry_col = 'industry'
    else:
        industry_col = None

    # 标准化（排除 industry 列）
    print("\n[2/6] 标准化...")
    scaler_cols = [c for c in feature_cols if c != 'industry']
    train_data[scaler_cols] = train_data[scaler_cols].replace([np.inf, -np.inf], np.nan)
    val_data[scaler_cols] = val_data[scaler_cols].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=scaler_cols)
    val_data = val_data.dropna(subset=scaler_cols)

    scaler = StandardScaler()
    train_data[scaler_cols] = scaler.fit_transform(train_data[scaler_cols])
    val_data[scaler_cols] = scaler.transform(val_data[scaler_cols])
    # 保持 industry 列不变（整数 0-30）
    joblib.dump(scaler, os.path.join(window_output_dir, 'scaler.pkl'))
    print(f"   特征维度: {len(scaler_cols)} + {'1 (industry)' if industry_col else '0 (no industry)'}")

    # 模型输入特征 = scaler_cols + (industry if present)
    model_feature_cols = scaler_cols + ([industry_col] if industry_col else [])

    # 构建数据集
    print("\n[3/6] 构建排序数据集...")
    # 训练集放宽未来窗口过滤（允许跨周末/节假日），样本量约提升5倍；
    # 验证集保持严格口径（自然日连续），与官方评测窗口及历史 final_score 可比
    span_days = config.get('max_future_span_days', 15)

    # 长历史窗口（2010起）物化全部序列会超出内存（~17MB/天样本），
    # 超过阈值自动切换懒加载数据集（窗口切片延迟到 __getitem__）
    lazy_threshold = config.get('lazy_dataset_threshold_days', 600)
    use_lazy = train_data['日期'].nunique() > lazy_threshold
    train_sampler = None
    if use_lazy:
        print(f"   训练天数 > {lazy_threshold}，使用懒加载数据集")
        train_dataset = build_lazy_ranking_dataset(
            train_data, model_feature_cols, config['sequence_length'],
            max_future_span_days=span_days)
        n_train = len(train_dataset)

        # ─── 时间衰减采样（应对长历史的非平稳性）───
        # 距训练截止日越近的样本被采样概率越高（指数衰减），
        # 远期历史提供正则化而非主导梯度；期望等价于损失逐样本加权
        half_life = config.get('time_decay_half_life_days')
        if half_life and n_train > 0:
            sample_dates = pd.to_datetime(pd.Series(train_dataset.sample_dates))
            age_days = (sample_dates.max() - sample_dates).dt.days.values.astype(np.float64)
            decay_weights = np.power(0.5, age_days / float(half_life))
            ess = decay_weights.sum() ** 2 / np.square(decay_weights).sum()  # 有效样本数
            print(f"   时间衰减采样: 半衰期 {half_life} 天, "
                  f"有效样本数 ESS≈{ess:.0f}/{n_train}, "
                  f"最老样本相对权重 {decay_weights.min():.4f}")
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.DoubleTensor(decay_weights),
                num_samples=n_train, replacement=True)
    else:
        train_sequences, train_targets, train_relevance, train_stock_indices, train_aux = \
            create_ranking_dataset_vectorized(train_data, model_feature_cols, config['sequence_length'],
                                              max_future_span_days=span_days)
        train_dataset = RankingDataset(train_sequences, train_targets, train_relevance,
                                       train_stock_indices, train_aux)
        n_train = len(train_sequences)

    val_data['日期_str'] = val_data['日期'].dt.strftime('%Y-%m-%d')
    val_sequences, val_targets, val_relevance, val_stock_indices, val_aux = \
        create_ranking_dataset_vectorized(
            val_data, model_feature_cols, config['sequence_length'],
            min_window_end_date=val_start
        )

    print(f"   训练样本: {n_train}, 验证样本: {len(val_sequences)}")

    if n_train == 0 or len(val_sequences) == 0:
        print("❌ 样本为空，跳过此窗口")
        return None

    # DataLoader
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance,
                                 val_stock_indices, val_aux)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\n[4/6] 设备: {device}")

    # 模型
    num_stocks = len(stockid2idx)
    # 模型输入维度：scaler_cols 数量（industry 由模型内部处理）
    model_input_dim = len(scaler_cols)
    # 选择模型类
    use_model = config.get('use_model', 'transformer')
    if use_model == 'lightweight':
        model = LightweightStockRanker(input_dim=model_input_dim, config=config, num_stocks=num_stocks)
    else:
        model = StockTransformer(input_dim=model_input_dim, config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"   参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5, temperature=1.0,
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
    aux_dir_criterion = nn.BCEWithLogitsLoss()
    aux_vol_criterion = nn.MSELoss()
    aux_return_criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

    # 学习率调度
    warmup_epochs = config.get('warmup_epochs', 5)
    warmup_start_lr = config.get('warmup_start_lr', 1e-6)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_start_lr / config['learning_rate'] + \
                   (1.0 - warmup_start_lr / config['learning_rate']) * epoch / warmup_epochs
        else:
            remaining = config['num_epochs'] - warmup_epochs
            progress = (epoch - warmup_epochs) / max(remaining, 1)
            return 1.0 - progress * 0.8

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练
    print(f"\n[5/6] 开始训练 (early_stopping_patience={config.get('early_stopping_patience', 10)})...")
    writer = SummaryWriter(log_dir=os.path.join(window_output_dir, 'log'))
    best_score = -float('inf')
    best_epoch = -1
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping_patience', 10)

    aux_dir_weight = config.get('aux_direction_weight', 0.1)
    aux_vol_weight = config.get('aux_volatility_weight', 0.1)
    aux_return_weight = config.get('aux_return_weight', 0.0)

    for epoch in range(config['num_epochs']):
        current_lr = scheduler.get_last_lr()[0]

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

        eval_loss, eval_metrics = evaluate_ranking_model(
            model, val_loader, criterion, device, writer, epoch
        )

        scheduler.step()

        current_final_score = eval_metrics.get('final_score', 0.0)
        if current_final_score > best_score:
            best_score = current_final_score
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(window_output_dir, 'best_model.pth'))
            with open(os.path.join(window_output_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        elif early_stopping_patience > 0:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  ⚠️ Early stopping at epoch {epoch+1} (best={best_score:.4f} @ epoch {best_epoch})")
                break

    writer.close()

    # 记录结果
    print(f"\n[6/6] Window {window_idx} 完成!")
    print(f"  Best epoch: {best_epoch}, Best final_score: {best_score:.4f}")

    with open(os.path.join(window_output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Window: {window_idx}\nBest epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")

    return {'score': best_score, 'output_dir': window_output_dir, 'best_epoch': best_epoch}


def run_walk_forward(config_name='light', seeds=None, data_path=None, output_base=None, window_indices=None,
                     data_start=None, half_life=None):
    """
    执行完整的 Walk-Forward 训练。

    Args:
        config_name: 配置名称 ('light', 'standard', 'v7')
        seeds: 随机种子列表，如 [42, 123]
        data_path: 数据目录路径
        output_base: 输出根目录
        window_indices: 只训练指定窗口（1-based，如 [1] 只跑 W1）；None = 全部窗口
    """
    if seeds is None:
        seeds = [42]

    if data_path is None:
        # 默认数据路径：项目根目录
        data_path = os.path.join(os.path.dirname(__file__), '..', '..')

    if output_base is None:
        output_base = r'C:\Users\huanx\Desktop\生产实习项目-沪深指数预测\model\walk_forward'

    os.makedirs(output_base, exist_ok=True)

    # 加载数据
    print("=" * 70)
    print(f"Walk-Forward Training — config={config_name}, seeds={seeds}")
    print("=" * 70)

    train_file = os.path.join(data_path, 'train.csv')
    if not os.path.exists(train_file):
        # 尝试当前目录
        train_file = os.path.join(os.path.dirname(__file__), '..', '..', 'train.csv')
    full_df = pd.read_csv(train_file)
    full_df['日期'] = pd.to_datetime(full_df['日期'])

    # ─── 截断实验：过滤指定日期之前的数据 ───────────
    if data_start:
        data_start_dt = pd.to_datetime(data_start)
        full_df = full_df[full_df['日期'] >= data_start_dt].copy()
        print(f"⚠️ 截断数据: 仅保留 {data_start} 之后 → {full_df.shape[0]:,} 行, "
              f"{full_df['日期'].min().date()} ~ {full_df['日期'].max().date()}")

    print(f"数据: {full_df.shape[0]:,} 行, {full_df['日期'].min().date()} ~ {full_df['日期'].max().date()}")

    # 建立股票索引
    all_stock_ids = sorted(full_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
    print(f"股票数: {len(stockid2idx)}")

    # 生成 Walk-Forward 窗口
    data_start = full_df['日期'].min()
    data_end = full_df['日期'].max()

    # 手动定义窗口（确保合理的训练数据量）
    # 6 个季末窗口 — 已验证为最优配置
    val_months = 2
    windows_raw = [
        ('2024-09-30', '2024-11-29'),   # W1:  train ~9个月
        ('2024-12-31', '2025-02-28'),   # W2:  train ~12个月
        ('2025-03-31', '2025-05-30'),   # W3:  train ~15个月
        ('2025-06-30', '2025-08-29'),   # W4:  train ~18个月
        ('2025-09-30', '2025-11-28'),   # W5:  train ~21个月
        ('2025-12-31', '2026-03-06'),   # W6:  train ~24个月
    ]

    # 过滤掉无效窗口（val_end 超出数据范围）
    windows = []
    for train_end_str, val_end_str in windows_raw:
        train_end = pd.to_datetime(train_end_str)
        val_end = pd.to_datetime(val_end_str)
        if val_end <= data_end and train_end >= data_start + pd.DateOffset(months=6):
            windows.append((train_end_str, val_end_str))

    print(f"\nWalk-Forward 窗口 ({len(windows)}):")
    for i, (te, ve) in enumerate(windows):
        print(f"  W{i+1}: train → {te}, val → {ve}")

    # 对每个 seed 训练
    all_results = {}
    for seed in seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED = {seed}")
        print(f"{'#'*70}")

        cfg = get_config_by_name(config_name)
        cfg['seed'] = seed
        cfg['data_path'] = data_path
        if half_life is not None:
            cfg['time_decay_half_life_days'] = half_life
            print(f"⚠️ 半衰期覆盖: {half_life} 天")

        seed_output_base = os.path.join(output_base, f'{config_name}_seed{seed}')
        os.makedirs(seed_output_base, exist_ok=True)

        seed_results = []
        for i, (train_end_str, val_end_str) in enumerate(windows):
            if window_indices is not None and (i + 1) not in window_indices:
                continue
            train_end = pd.to_datetime(train_end_str)
            val_end = pd.to_datetime(val_end_str)
            val_start = train_end + pd.tseries.offsets.BDay(1)

            # 序列上下文：验证集需要前60个交易日的历史
            val_context_start = val_start - pd.tseries.offsets.BDay(cfg['sequence_length'] - 1)

            # 切分数据
            train_df = full_df[full_df['日期'] <= train_end].copy()
            val_df = full_df[(full_df['日期'] >= val_context_start) &
                           (full_df['日期'] <= val_end)].copy()

            train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
            val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

            print(f"\n--- Window {i+1}/{len(windows)} (seed={seed}) ---")
            print(f"  Train: {train_df['日期'].min()} ~ {train_df['日期'].max()} "
                  f"({train_df['日期'].nunique()} 天)")
            print(f"  Val:   {val_df['日期'].min()} ~ {val_df['日期'].max()} "
                  f"({val_df['日期'].nunique()} 天, context from {val_context_start.date()})")

            result = train_single_window(
                train_df, val_df, val_start.strftime('%Y-%m-%d'),
                cfg, stockid2idx, i + 1, seed_output_base
            )
            if result:
                result['window_train_end'] = train_end_str
                result['window_val_end'] = val_end_str
                seed_results.append(result)

        all_results[f'seed{seed}'] = seed_results

        # 汇总该 seed 在各窗口的表现
        scores = [r['score'] for r in seed_results]
        mean_score = np.mean(scores) if scores else 0
        print(f"\nSeed {seed} 完成! {len(seed_results)}/{len(windows)} 窗口成功")
        print(f"  各窗口 final_score: {[f'{s:.4f}' for s in scores]}")
        print(f"  平均 final_score:    {mean_score:.4f}")

    # 保存汇总结果
    summary = {
        'config': config_name,
        'seeds': seeds,
        'windows': windows,
        'results': {
            seed_key: [
                {k: v for k, v in r.items() if k != 'output_dir'}
                for r in results
            ]
            for seed_key, results in all_results.items()
        },
        'timestamp': datetime.now().isoformat(),
    }
    summary_path = os.path.join(output_base, 'walk_forward_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📊 汇总保存到: {summary_path}")

    return all_results


# ══════════════════════════════════════════════════════════
# Walk-Forward 集成预测
# ══════════════════════════════════════════════════════════

def load_walk_forward_models(output_base, config_name='light', seeds=None, device=None):
    """加载所有 Walk-Forward 窗口的模型"""
    if seeds is None:
        seeds = [42]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = []
    for seed in seeds:
        seed_dir = os.path.join(output_base, f'{config_name}_seed{seed}')
        if not os.path.exists(seed_dir):
            continue
        for window_dir_name in sorted(os.listdir(seed_dir)):
            if not window_dir_name.startswith('window_'):
                continue
            window_dir = os.path.join(seed_dir, window_dir_name)
            model_path = os.path.join(window_dir, 'best_model.pth')
            config_path = os.path.join(window_dir, 'config.json')
            scaler_path = os.path.join(window_dir, 'scaler.pkl')

            if not all(os.path.exists(p) for p in [model_path, config_path, scaler_path]):
                continue

            with open(config_path) as f:
                cfg = json.load(f)
            scaler = joblib.load(scaler_path)

            # 根据配置选择模型类
            use_model = cfg.get('use_model', 'transformer')
            if use_model == 'lightweight':
                model = LightweightStockRanker(
                    input_dim=scaler.n_features_in_,
                    config=cfg,
                    num_stocks=300
                )
            else:
                model = StockTransformer(
                    input_dim=scaler.n_features_in_,
                    config=cfg,
                    num_stocks=300
                )
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True),
                                strict=False)
            model.to(device)
            model.eval()
            models.append({
                'model': model,
                'scaler': scaler,
                'config': cfg,
                'window_dir': window_dir,
                'seed': seed,
            })
    return models


def walk_forward_predict(models, processed_df, feature_cols, stock_codes, pred_date, device):
    """
    用所有 Walk-Forward 模型进行集成预测。

    使用 median 聚合各模型的分数（对异常值更鲁棒），
    同时计算多模型排序一致性用于过滤。
    """
    seq_len = models[0]['config']['sequence_length'] if models else 60

    # 构建预测序列
    sequences = []
    valid_stocks = []
    for code in stock_codes:
        hist = processed_df[
            (processed_df['股票代码'] == code) &
            (processed_df['日期'] <= pred_date)
        ].sort_values('日期').tail(seq_len)
        if len(hist) == seq_len:
            seq = hist[feature_cols].values.astype(np.float32)
            sequences.append(seq)
            valid_stocks.append(code)

    if not sequences:
        return None

    seq_tensor = torch.FloatTensor(np.array(sequences)).unsqueeze(0).to(device)

    all_scores = []
    all_predictions = []  # 每个模型的 Top-10 列表

    for m in models:
        with torch.no_grad():
            scores, aux = m['model'](seq_tensor, return_aux=True)
            scores_np = scores.squeeze(0).cpu().numpy()
            all_scores.append(scores_np)

            # 记录每个模型的 Top-10
            top10_idx = np.argsort(scores_np)[-10:][::-1]
            top10_codes = set(valid_stocks[i] for i in top10_idx)
            all_predictions.append(top10_codes)

    # Median 聚合（对异常窗口鲁棒）
    scores_median = np.median(all_scores, axis=0)

    # ─── P5: 一致性过滤 ───
    from collections import Counter
    n_models = len(models)
    consistency_counts = Counter()
    for pred_set in all_predictions:
        consistency_counts.update(pred_set)

    # 至少在 50% 的模型中出现在 Top-10
    min_consensus = max(2, int(n_models * 0.4))
    consistent_stocks = {code for code, cnt in consistency_counts.items()
                         if cnt >= min_consensus}

    print(f"  一致性过滤: {len(consistent_stocks)}/{len(valid_stocks)} 只股票通过 "
          f"(≥{min_consensus}/{n_models} 模型共识)")

    # 构建结果
    results = []
    for i, code in enumerate(valid_stocks):
        results.append({
            '股票代码': code,
            '预测分数': float(scores_median[i]),
            '一致性': consistency_counts.get(code, 0),
            '模型共识数': consistency_counts.get(code, 0),
            '通过一致性': code in consistent_stocks,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('预测分数', ascending=False).reset_index(drop=True)

    # 只保留通过一致性过滤的股票（如果足够多的话）
    if len(consistent_stocks) >= 5:
        results_df = results_df[results_df['通过一致性']].reset_index(drop=True)

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward 训练与预测')
    parser.add_argument('--config', type=str, default='light',
                        choices=['light', 'standard', 'v7', 'v8_improved', 'lightweight'],
                        help='模型配置名称')
    parser.add_argument('--seeds', type=str, default='42',
                        help='随机种子，逗号分隔 (如 42,123,456)')
    parser.add_argument('--data-path', type=str, default=None,
                        help='数据目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出根目录')
    parser.add_argument('--predict-only', action='store_true',
                        help='仅预测（不训练）')
    parser.add_argument('--windows', type=str, default=None,
                        help='只训练指定窗口，逗号分隔的1-based序号 (如 1 或 1,3,6)；默认全部')
    parser.add_argument('--data-start', type=str, default=None,
                        help='训练数据起始日期 (YYYY-MM-DD)，过滤此前数据用于截断实验')
    parser.add_argument('--half-life', type=int, default=None,
                        help='时间衰减半衰期（天），覆盖config中的time_decay_half_life_days')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    window_indices = ([int(w.strip()) for w in args.windows.split(',')]
                      if args.windows else None)

    if not args.predict_only:
        mp.set_start_method('spawn', force=True)
        run_walk_forward(
            config_name=args.config,
            seeds=seeds,
            data_path=args.data_path,
            output_base=args.output,
            window_indices=window_indices,
            data_start=args.data_start,
            half_life=args.half_life,
        )


if __name__ == '__main__':
    main()

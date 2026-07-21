"""
test_integration.py — 端到端集成烟雾测试

最小链路：合成数据 → 数据集构建 → 1 epoch 训练 → 预测 → 验证输出形状。
不依赖真实 CSV 文件、GPU 或预训练模型。运行时间 < 30 秒。
"""
import numpy as np
import torch
import pytest
import tempfile
import os


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

INPUT_DIM = 204


def _make_minimal_config():
    """最小训练配置，用于快速集成测试"""
    return {
        'sequence_length': 15,
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1,
        'dim_feedforward': 64,
        'dropout': 0.1,
        'feature_num': '158+39',
        'use_tcn': False,
        'use_feature_interaction': False,
        'use_market_aggregation': True,
        'market_dim': 16,
        'market_pool_heads': 2,
        'use_calendar_features': False,
        'use_fundamentals': False,
        'use_momentum_features': False,
        'use_macro_features': False,
        'use_industry_embedding': False,
        # 训练参数
        'learning_rate': 1e-4,
        'batch_size': 4,
        'num_epochs': 1,
        'data_augment_prob': 0.0,  # 关闭增强以加速
        # 损失参数
        'pairwise_weight': 1.0,
        'ndcg_weight': 0.3,
        'precision_weight': 0.5,
        'aux_dir_weight': 0.0,
        'aux_vol_weight': 0.0,
        'aux_return_weight': 0.0,
        'portfolio_weight': 0.0,
        'market_loss_weight': 0.0,
        'weight_factor': 3.0,
        'temperature': 1.0,
        'k': 5,
        'use_mixed_label': False,
        'use_quantile_label': True,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 5,
    }


def _make_synthetic_raw_df(num_dates=30, num_stocks=20):
    """生成模拟的原始股票日线数据（含价格和技术指标）"""
    import pandas as pd

    np.random.seed(42)
    rows = []
    base_date = pd.Timestamp('2025-01-02')

    for d in range(num_dates):
        date = base_date + pd.DateOffset(days=d)
        for s in range(num_stocks):
            stock_code = f'{600000 + s:06d}'
            price = 10.0 + s * 2.0 + np.random.randn() * 0.5
            row = {
                '日期': date,
                '股票代码': stock_code,
                'instrument': s,
                '开盘': price,
                '收盘': price + np.random.randn() * 0.2,
                '最高': price + abs(np.random.randn()) * 0.3,
                '最低': price - abs(np.random.randn()) * 0.3,
                '成交量': abs(np.random.randn()) * 1e6 + 1e6,
                '成交额': abs(np.random.randn()) * 1e8 + 1e8,
                '振幅': abs(np.random.randn()) * 0.03,
                '涨跌额': np.random.randn() * 0.1,
                '换手率': abs(np.random.randn()) * 0.02,
                '涨跌幅': np.random.randn() * 0.02,
            }
            # 添加 39 个技术指标列（随机值）
            for col in ['sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi',
                        'macd', 'macd_signal', 'volume_change', 'obv',
                        'volume_ma_5', 'volume_ma_20', 'volume_ratio',
                        'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
                        'atr_14', 'ema_60', 'volatility_10', 'volatility_20',
                        'return_1', 'return_5', 'return_10',
                        'high_low_spread', 'open_close_spread',
                        'high_close_spread', 'low_close_spread']:
                row[col] = np.random.randn() * 0.01

            # 添加 158 Alpha 因子（随机值，用简单占位符）
            for i in range(1, 159):
                col = f'alpha_{i}'
                row[col] = np.random.randn() * 0.01

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

def test_end_to_end_synthetic_data_training(tmp_path):
    """端到端最小链路：数据构建 → 模型创建 → 1 epoch 训练 → 预测"""
    from model import StockTransformer
    from train import (
        _build_label_and_clean, WeightedRankingLoss, RankingDataset,
        collate_fn, train_ranking_model
    )
    from utils import create_ranking_dataset_vectorized
    from sklearn.preprocessing import StandardScaler

    # ── 1. 生成合成数据 ──
    df = _make_synthetic_raw_df(num_dates=30, num_stocks=20)
    config = _make_minimal_config()

    # ── 2. 标签构建 ──
    processed = _build_label_and_clean(
        df, drop_small_open=False, use_quantile_label=True,
        use_mixed_label=False
    )
    assert len(processed) > 0, "标签构建不应返回空 DataFrame"
    assert 'label' in processed.columns

    # ── 3. 特征列 ──
    feature_cols = [c for c in df.columns if c not in (
        '股票代码', '日期', 'label', 'label_abs', 'instrument'
    )]
    # 限制为实际存在的列
    feature_cols = [c for c in feature_cols if c in processed.columns]

    # ── 4. 标准化 ──
    scaler = StandardScaler()
    processed[feature_cols] = scaler.fit_transform(processed[feature_cols].values)

    # ── 5. 数据集构建 ──
    result = create_ranking_dataset_vectorized(
        processed, feature_cols, config['sequence_length'],
        max_future_span_days=15
    )
    sequences, targets, relevance, stock_indices = result[:4]
    aux_labels = result[4] if len(result) > 4 else None
    assert len(sequences) > 0, "数据集构建应产生至少一个样本"

    # ── 6. 创建 DataLoader ──
    dataset = RankingDataset(sequences, targets, relevance, stock_indices, aux_labels)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, batch_size=min(config['batch_size'], len(dataset)),
        shuffle=True, collate_fn=collate_fn
    )

    # ── 7. 创建模型 ──
    model = StockTransformer(input_dim=len(feature_cols), config=config, num_stocks=20)
    device = torch.device('cpu')
    model = model.to(device)

    # ── 8. 损失函数 + 优化器 ──
    criterion = WeightedRankingLoss(
        temperature=config['temperature'], k=config['k'],
        weight_factor=config['weight_factor'],
        pairwise_weight=config['pairwise_weight'],
        ndcg_weight=config['ndcg_weight'],
        precision_weight=config['precision_weight'],
        portfolio_weight=config['portfolio_weight'],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # ── 9. 训练 1 epoch ──
    model.train()
    train_ranking_model(
        model, dataloader, criterion, optimizer, device,
        epoch=0, writer=None, config=config,
        aux_dir_weight=0.0, aux_vol_weight=0.0, aux_return_weight=0.0,
    )

    # ── 10. 预测验证 ──
    model.eval()
    # 取第一个样本做预测
    sample = dataset[0]
    seq = sample['sequences'].unsqueeze(0)  # [1, N, T, F]
    with torch.no_grad():
        scores = model(seq, return_aux=False)
    assert scores.shape[0] == 1, f"预测 batch size 应为 1, 实际 {scores.shape[0]}"
    assert scores.shape[1] > 0, "预测应返回所有股票的分数"
    assert not torch.isnan(scores).any(), "预测分数不应含 NaN"

    # ── 11. 验证分数可用 ──
    scores_np = scores.squeeze(0).numpy()
    top_k = min(5, len(scores_np))
    top_indices = np.argsort(scores_np)[::-1][:top_k]
    assert len(top_indices) == top_k, f"Top-K 应选出 {top_k} 只股票"


def test_end_to_end_prediction_pipeline(tmp_path):
    """验证训练后的模型保存/加载链路"""
    from model import StockTransformer
    import joblib
    import json

    config = _make_minimal_config()
    model = StockTransformer(input_dim=INPUT_DIM, config=config, num_stocks=20)

    # 保存
    save_dir = tmp_path / "test_model"
    save_dir.mkdir()
    torch.save(model.state_dict(), save_dir / "best_model.pth")
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config, f)
    scaler = __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler()
    scaler.fit(np.random.randn(100, INPUT_DIM))
    joblib.dump(scaler, save_dir / "scaler.pkl")

    # 加载
    with open(save_dir / "config.json") as f:
        loaded_config = json.load(f)
    loaded_scaler = joblib.load(save_dir / "scaler.pkl")
    loaded_model = StockTransformer(
        input_dim=loaded_scaler.n_features_in_,
        config=loaded_config, num_stocks=20
    )
    loaded_model.load_state_dict(
        torch.load(save_dir / "best_model.pth", map_location='cpu', weights_only=True),
        strict=False
    )
    loaded_model.eval()

    # 验证前向传播仍正常
    x = torch.randn(1, 20, config['sequence_length'], INPUT_DIM)
    with torch.no_grad():
        scores = loaded_model(x, return_aux=False)
    assert scores.shape == (1, 20)

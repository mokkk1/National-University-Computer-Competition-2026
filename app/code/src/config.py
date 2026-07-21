# V7 配置 — 修复版：恢复正则化 + 绝对收益预测 + 收益门控
# 问题诊断: V6 关闭了辅助任务/数据增强导致过拟合动量信号
# 修复: 恢复正则化 + 新增 return_head + 收益门控后处理

import os

# 项目根目录（自动推断，兼容 Windows / Linux / Docker）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sequence_length = 60
feature_num = '158+39+fundamental+momentum'
config = {
    'sequence_length': sequence_length,

    # ─── 模型（恢复容量，靠正则化防过拟合）─────
    'd_model': 256,           # 恢复 256
    'nhead': 4,
    'num_layers': 3,          # 恢复 3
    'dim_feedforward': 512,   # 恢复 512
    'batch_size': 4,
    'num_epochs': 50,         # 50 epoch
    'learning_rate': 1e-5,
    'dropout': 0.15,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    # ─── 损失函数 ──────────────────────────────
    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 3.0,
    'ndcg_weight': 0.3,
    'precision_weight': 0.5,
    'use_exact_lambda': True,
    'use_gumbel_ndcg': False,

    # ─── TCN ────────────────────────────────────
    'use_tcn': True,
    'tcn_kernel_sizes': [3, 5, 7],
    'tcn_dropout': 0.1,

    # ─── 特征交互 ──────────────────────────────
    'use_feature_interaction': True,
    'fi_rank': 64,            # 恢复 64

    # ─── 辅助任务（重新启用，提供正则化）───────
    'aux_direction_weight': 0.1,
    'aux_volatility_weight': 0.1,
    'aux_return_weight': 0.3,          # 新增：绝对收益回归 (Huber loss)

    # ─── 基本面特征 ─────────────────────────────
    'use_fundamentals': True,

    # ─── 动量特征 ──────────────────────────────
    'use_momentum_features': True,
    'momentum_features': ['ret5', 'ret20', 'vol20', 'sharpe5'],

    # ─── 数据增强（重新启用，适度）─────────────
    'augment_prob': 0.4,
    'time_mask_ratio': 0.1,
    'feature_noise_std': 0.003,
    'stock_dropout_ratio': 0.15,

    # ─── 训练策略 ──────────────────────────────
    'warmup_epochs': 5,
    'warmup_start_lr': 1e-6,
    'early_stopping_patience': 12,
    'val_months': 3,

    # ─── 后处理（收益门控）─────────────────────
    'post_top_k': 10,
    'use_volatility_penalty': True,
    'use_return_gate': True,             # 新增：启用收益门控
    'min_return_threshold': 0.0,         # 新增：只选预测正收益的股票
    'return_gate_fallback': 'equal',     # 新增：无股票通过门控时回退等权

    # ─── P1 改进：标签策略 ──────────────────
    'use_quantile_label': True,          # True=分位数rank标签
    'use_mixed_label': True,             # 🌟 P1: 混合标签 (rank + abs return)
    'mixed_label_alpha': 0.7,            # 混合标签中rank权重 (0~1)

    # ─── 长历史训练（2010起）───────────────
    'max_future_span_days': 15,          # 训练窗口未来5日的自然日跨度上限（放宽过滤）
    'time_decay_half_life_days': 730,    # 时间衰减采样半衰期（自然日）：距训练截止日每2年权重减半，
                                         # 让远期历史提供正则化而非主导梯度；None=均匀采样

    # ─── P1 改进：Portfolio Loss ─────────────
    'portfolio_loss_weight': 0.15,       # Portfolio return loss 权重
    'portfolio_temperature': 0.5,        # Gumbel-softmax 温度

    # ─── ★ 市场聚合架构 (P1 核心改进) ──────────
    'use_market_aggregation': True,      # 启用市场聚合+门控+方向预测
    'market_dim': 64,                    # 市场状态向量维度
    'market_pool_heads': 4,              # 市场注意力池化头数
    'market_loss_weight': 0.2,           # 市场方向预测损失权重

    # ─── 日历特征（❌ 已验证有害，禁用）───────
    'use_calendar_features': False,      # ❌ days_to_qe/is_qe_month 让非季末日退化4.39pp

    # ─── P0 改进：市场门控后处理 ─────────────
    'use_market_gate': True,             # 启用市场环境自适应选股

    'seed': 42,
    'ensemble_seeds': [42, 123, 456],

    'output_dir': os.path.join(_PROJECT_ROOT, 'model', f'60_{feature_num}_v8_improved'),
    'data_path': _PROJECT_ROOT,
    'fundamental_path': os.path.join(_PROJECT_ROOT, 'data', 'fundamentals.csv'),
}


# ═══════════════════════════════════════════════════════════════
# ★ V9 轻量级配置 (方案1+2: 统计特征 + GRU + 方向分类)
# ═══════════════════════════════════════════════════════════════

LIGHTWEIGHT_CONFIG = {
    'sequence_length': 60,
    'd_model': 128,              # 从 256 削减
    'gru_hidden': 48,            # GRU 隐藏维度
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 1e-4,       # 模型更小，可以用更大的 LR
    'dropout': 0.2,              # 更强的 dropout 防过拟合
    'feature_num': '158+39+fundamental+momentum',
    'max_grad_norm': 5.0,

    # ─── 损失函数 ──────────────────────────────
    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 3.0,
    'ndcg_weight': 0.3,
    'precision_weight': 0.5,
    'use_exact_lambda': True,
    'use_gumbel_ndcg': False,

    # ─── 方案1: 轻量架构 (去掉 TCN/特征交互/市场聚合) ────
    'use_model': 'lightweight',   # ★ 使用 LightweightStockRanker
    'use_tcn': False,
    'use_feature_interaction': False,
    'use_market_aggregation': False,  # 模型内部用简单均值池化
    'use_calendar_features': False,

    # ─── 方案2: 方向分类替代回归 ──────────────────
    'aux_direction_weight': 0.5,    # ★ 从 0.1 提升到 0.5 (主辅助任务)
    'aux_volatility_weight': 0.1,
    'aux_return_weight': 0.05,      # ★ 从 0.3 降到 0.05 (弱训练，仅用于门控)
    'market_loss_weight': 0.3,      # 市场方向 BCE

    # ─── 基本面 + 动量 ─────────────────────────────
    'use_fundamentals': True,
    'use_momentum_features': True,
    'momentum_features': ['ret5', 'ret20', 'vol20', 'sharpe5'],

    # ─── 宏观 + 行业 ───────────────────────────────
    'use_macro_features': True,
    'use_industry_embedding': True,
    'num_industries': 31,
    'industry_emb_dim': 8,          # 从 16 削减

    # ─── 数据增强 (适度) ─────────────────────────
    'augment_prob': 0.3,
    'time_mask_ratio': 0.05,
    'feature_noise_std': 0.002,
    'stock_dropout_ratio': 0.1,

    # ─── 训练策略 ──────────────────────────────
    'warmup_epochs': 3,
    'warmup_start_lr': 1e-5,
    'early_stopping_patience': 15,  # 更长的 patience，模型更小需要更多 epoch
    'val_months': 2,

    # ─── 标签策略 ──────────────────────────────
    'use_quantile_label': True,
    'use_mixed_label': True,
    'mixed_label_alpha': 0.7,

    # ─── Portfolio Loss (保留) ──────────────────
    'portfolio_loss_weight': 0.15,
    'portfolio_temperature': 0.5,

    # ─── 后处理 ──────────────────────────────
    'post_top_k': 10,
    'use_volatility_penalty': True,
    'use_return_gate': True,
    'min_return_threshold': 0.0,
    'return_gate_fallback': 'equal',
    'use_market_gate': True,

    'seed': 42,
    'ensemble_seeds': [42, 123, 456],

    'output_dir': os.path.join(_PROJECT_ROOT, 'model', 'lightweight_v9'),
    'data_path': _PROJECT_ROOT,
    'fundamental_path': os.path.join(_PROJECT_ROOT, 'data', 'fundamentals.csv'),
}

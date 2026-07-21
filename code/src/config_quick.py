# Quick test config
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = {
    "sequence_length": 60,
    "d_model": 256,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 512,
    "batch_size": 4,
    "num_epochs": 40,
    "learning_rate": 1e-05,
    "dropout": 0.15,
    "feature_num": "158+39",
    "max_grad_norm": 5.0,
    "pairwise_weight": 1,
    "base_weight": 1.0,
    "top5_weight": 3.0,
    "ndcg_weight": 0.3,
    "precision_weight": 0.5,
    "use_exact_lambda": true,
    "use_gumbel_ndcg": false,
    "use_tcn": true,
    "tcn_kernel_sizes": [
        3,
        5,
        7
    ],
    "tcn_dropout": 0.1,
    "use_feature_interaction": true,
    "fi_rank": 64,
    "aux_direction_weight": 0.1,
    "aux_volatility_weight": 0.1,
    "aux_return_weight": 0.3,           # 绝对收益回归损失权重（新增）
    "use_return_gate": True,             # 启用收益门控后处理（新增）
    "min_return_threshold": 0.0,         # 收益门控阈值，>0 才入选（新增）
    "return_gate_fallback": "equal",     # 无股票通过门控时的回退策略（新增）
    "use_fundamentals": true,
    "fundamental_features": [
        "pe",
        "pb",
        "ps",
        "roe",
        "roa",
        "gross_margin",
        "revenue_yoy",
        "profit_yoy",
        "north_holding_pct",
        "fund_flow_5d"
    ],
    "augment_prob": 0.5,
    "time_mask_ratio": 0.15,
    "feature_noise_std": 0.005,
    "stock_dropout_ratio": 0.2,
    "label_smoothing": 0.05,
    "ensemble_seeds": [
        42,
        123,
        456
    ],
    "ensemble_mode": "weighted",
    "multi_period_dates": 5,
    "warmup_epochs": 5,
    "warmup_start_lr": 1e-06,
    "early_stopping_patience": 10,
    "post_top_k": 10,
    "use_volatility_penalty": true,
    "seed": 42,
    "output_dir": os.path.join(_PROJECT_ROOT, "model", "60_158+39_v2"),
    "data_path": _PROJECT_ROOT
}

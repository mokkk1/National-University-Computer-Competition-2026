"""
GBDT Stacking 原型：Transformer 表征 + LightGBM LambdaRank

流程:
1. 加载 V7 已训练的 StockTransformer，冻结权重
2. 对训练集每一天，用 Transformer 提取 128 维 ranking_features
3. 拼接原始特征最后一天值 (211 维) → 339 维
4. 逐日构建 LightGBM LambdaRank group，训练
5. 测试集上预测 → Top-5 选股 → 计算收益率并与 V7 对比
"""

import os, sys, json, math, random
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ─── 路径配置 ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "60_158+39+fundamental+momentum_v7")
DATA_DIR = PROJECT_ROOT   # train.csv / test.csv 在项目根目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "gbdt_stacking")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 添加 code/src 到 path
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "src"))

from model import StockTransformer
from train import (
    set_seed, preprocess_data, preprocess_val_data,
    feature_cloums_map, feature_engineer_func_map,
    _build_label_and_clean, RankingDataset, collate_fn,
    augment_batch
)
from utils import create_ranking_dataset_vectorized
from fundamental import load_fundamentals


def add_extract_method(model):
    """猴子补丁：给模型加一个 extract_features 方法，返回 ranking_features"""
    def extract_features(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)
        if self.use_tcn:
            src_proj = self.multi_scale_conv(src_proj)
        temporal_features = self.temporal_encoder(src_proj)
        aggregated_features = self.feature_attention(temporal_features)
        if self.use_feature_interaction:
            aggregated_features = self.feature_interaction(aggregated_features)
        stock_features = aggregated_features.view(batch_size, num_stocks, -1)
        interactive_features = self.cross_stock_attention(stock_features)
        interactive_features = interactive_features.view(batch_size * num_stocks, -1)
        ranking_features = self.ranking_layers(interactive_features)  # [B*N, 128]
        # reshape 回 [B, N, 128]
        ranking_features = ranking_features.view(batch_size, num_stocks, -1)
        return ranking_features
    model.extract_features = extract_features.__get__(model, type(model))


# ══════════════════════════════════════════════════════════
# Step 1: 加载 V7 模型 & 数据预处理
# ══════════════════════════════════════════════════════════

def load_v7_model(device):
    """加载 V7 训练好的模型和配置"""
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        config = json.load(f)

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    # 加载数据获取 num_stocks
    data_file = os.path.join(DATA_DIR, "train.csv")
    full_df = pd.read_csv(data_file)
    all_stock_ids = full_df["股票代码"].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    # 确定特征维度
    feature_columns = feature_cloums_map[config["feature_num"]]
    input_dim = len(feature_columns)

    # 初始化模型并加载权重
    model = StockTransformer(input_dim=input_dim, config=config, num_stocks=num_stocks)
    state_dict = torch.load(
        os.path.join(MODEL_DIR, "best_model.pth"),
        map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 加 extract_features 方法
    add_extract_method(model)

    print(f"V7 模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"特征数: {input_dim}, 股票数: {num_stocks}")
    return model, config, scaler, stockid2idx, feature_columns


def prepare_data(config, stockid2idx):
    """运行特征工程，返回处理后的 train / val / test DataFrame"""
    data_file = os.path.join(DATA_DIR, "train.csv")
    full_df = pd.read_csv(data_file)

    from train import split_train_val_by_last_month
    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df, config["sequence_length"], config.get("val_months", 3)
    )

    # 基本面数据
    if config.get("use_fundamentals", False):
        print("加载基本面数据...")
        load_fundamentals()

    # 特征工程
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    # 加载 test.csv
    test_file = os.path.join(DATA_DIR, "test.csv")
    test_df = pd.read_csv(test_file)
    # 合并 train+test 给测试集提供序列上下文
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined_data, _ = preprocess_data(combined, is_train=False, stockid2idx=stockid2idx)

    # 标准化
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    combined_data[features] = combined_data[features].replace([np.inf, -np.inf], np.nan)

    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    combined_data = combined_data.dropna(subset=features)

    scaler = StandardScaler()
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    combined_data[features] = scaler.transform(combined_data[features])

    print(f"训练集: {len(train_data)} 行, 验证集: {len(val_data)} 行")
    return train_data, val_data, combined_data, features, scaler


# ══════════════════════════════════════════════════════════
# Step 2: Transformer 特征提取
# ══════════════════════════════════════════════════════════

def extract_transformer_features(model, data, features, sequence_length, stockid2idx, device):
    """
    对 data 中每一天，用 Transformer 提取每只股票的 128 维 ranking_features
    同时提取原始特征最后一天的值 (211 维)

    返回:
        day_records: [{'date': str, 'stock_code': str, 'label': float, 'label_abs': float,
                       'tf_feat': np.array(128,), 'raw_feat': np.array(211,)}]
    """
    model.eval()
    data = data.copy()
    data["日期"] = pd.to_datetime(data["日期"])
    data = data.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    unique_dates = sorted(data["日期"].unique())
    day_records = []

    # 每天一个 batch
    for target_date in tqdm(unique_dates, desc="提取 Transformer 特征"):
        target_date = pd.Timestamp(target_date)
        day_data = data[data["日期"] == target_date]

        day_sequences = []
        day_stock_codes = []
        day_labels = []
        day_labels_abs = []

        for _, row in day_data.iterrows():
            stock_code = row["股票代码"]
            # 取该股票在 target_date 及之前的 60 天历史
            stock_history = data[
                (data["股票代码"] == stock_code) &
                (data["日期"] <= target_date)
            ].sort_values("日期").tail(sequence_length)

            if len(stock_history) < sequence_length:
                continue

            seq = stock_history[features].values.astype(np.float32)
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_labels.append(row.get("label", np.nan))
            day_labels_abs.append(row.get("label_abs", np.nan))

        if len(day_sequences) < 5:
            continue

        # [1, N, 60, 211]
        seq_tensor = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)

        with torch.no_grad():
            tf_features = model.extract_features(seq_tensor)  # [1, N, 128]

        tf_features = tf_features.squeeze(0).cpu().numpy()  # [N, 128]

        # 原始特征最后一天的值
        raw_last_day = np.array([s[-1, :] for s in day_sequences])  # [N, 211]

        for i, stock_code in enumerate(day_stock_codes):
            day_records.append({
                "date": str(target_date.date()),
                "stock_code": stock_code,
                "label": day_labels[i],
                "label_abs": day_labels_abs[i],
                "tf_feat": tf_features[i],       # 128 dims
                "raw_feat": raw_last_day[i],     # 211 dims
            })

    print(f"提取完成: {len(day_records)} 条样本, {len(unique_dates)} 个交易日")
    return day_records


# ══════════════════════════════════════════════════════════
# Step 3: 构建 LightGBM 数据集
# ══════════════════════════════════════════════════════════

def continuous_to_relevance(labels, n_levels=10):
    """
    将连续超额收益转为离散相关度等级 (0 ~ n_levels-1)。
    每天组内按分位数量化，LambdaRank 需要整数标签计算 NDCG。
    """
    if len(labels) < n_levels:
        # 股票太少时直接排序赋等级
        ranks = np.argsort(np.argsort(labels))  # 0 = 最低收益
        # 映射到 0 ~ n_levels-1
        relevance = np.clip(np.floor(ranks / max(1, len(labels)) * n_levels), 0, n_levels-1).astype(int)
        return relevance

    # 按分位数分配等级
    relevance = np.zeros(len(labels), dtype=int)
    for level in range(1, n_levels):
        threshold = np.quantile(labels, level / n_levels)
        relevance[labels > threshold] = level
    # Top 5 单独标记为最高等级（强化 Top-K 信号）
    top5_idx = np.argsort(labels)[-5:]
    relevance[top5_idx] = n_levels - 1
    return relevance


def build_gbdt_dataset(records, tf_dim=128, raw_dim=211, n_relevance_levels=10):
    """
    将 records 转换为 LightGBM LambdaRank 所需格式: X, y (整数相关度), group

    每个交易日 = 一个 query group
    标签从连续超额收益转为组内分位数等级（0~9）
    """
    records = sorted(records, key=lambda r: r["date"])
    dates = sorted(set(r["date"] for r in records))

    X_list, y_cont_list, group_records = [], [], []

    for date in tqdm(dates, desc="构建 GBDT 数据集"):
        day_records = [r for r in records if r["date"] == date]
        if len(day_records) < 5:
            continue
        group_records.append(day_records)

    # 按天独立做分位数离散化
    X_list, y_list, group_list = [], [], []
    for day_records in group_records:
        labels = np.array([r["label"] for r in day_records])
        relevance = continuous_to_relevance(labels, n_levels=n_relevance_levels)

        for r, rel in zip(day_records, relevance):
            feat = np.concatenate([r["tf_feat"], r["raw_feat"]])  # 128 + 219 = 347
            X_list.append(feat)
            y_list.append(rel)

        group_list.append(len(day_records))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"GBDT 数据集: X={X.shape}, y={y.shape} (相关度 0~{n_relevance_levels-1}), groups={len(group_list)}")
    # 打印等级分布
    unique, counts = np.unique(y, return_counts=True)
    print(f"  标签分布: {dict(zip(unique, counts))}")
    return X, y, group_list


# ══════════════════════════════════════════════════════════
# Step 4: 训练 LightGBM LambdaRank
# ══════════════════════════════════════════════════════════

def train_gbdt(X_train, y_train, groups_train, X_val, y_val, groups_val):
    """训练 LightGBM LambdaRank 模型"""
    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "boosting_type": "gbdt",
        "num_leaves": 128,
        "max_depth": 10,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 30,
        "num_iterations": 2000,
        "early_stopping_rounds": 150,
        "verbose": 100,
        "seed": 42,
        "num_threads": 16,
        "force_col_wise": True,   # 样本数 < 特征数时更稳定
    }

    print("\n开始训练 LightGBM LambdaRank...")
    print(f"  训练样本: {X_train.shape[0]}, 特征: {X_train.shape[1]}")
    print(f"  训练 query groups: {len(groups_train)}, 验证 query groups: {len(groups_val)}")

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(150),
            lgb.log_evaluation(100),
        ],
    )

    # 特征重要性
    print(f"\nTop-10 特征重要性 (gain):")
    importance = model.feature_importance(importance_type="gain")
    feature_names = [f"tf_{i}" for i in range(128)] + [f"raw_{i}" for i in range(211)]
    sorted_idx = np.argsort(importance)[::-1][:10]
    for i in sorted_idx:
        print(f"  {feature_names[i]:15s}: {importance[i]:.0f}")

    return model


# ══════════════════════════════════════════════════════════
# Step 5: 测试集预测 & 收益率计算
# ══════════════════════════════════════════════════════════

def predict_and_evaluate(gbdt_model, test_records, test_csv_path, top_k=5):
    """
    在测试集最后一天预测 Top-K 股票，计算真实 5 日收益率
    """
    test_dates = sorted(set(r["date"] for r in test_records))
    last_date = test_dates[-1]
    print(f"\n测试集日期范围: {test_dates[0]} ~ {last_date}")

    # 最后一天的样本
    last_day_records = [r for r in test_records if r["date"] == last_date]
    if len(last_day_records) < top_k:
        print(f"⚠️ 最后一天只有 {len(last_day_records)} 只股票，不足 {top_k}")
        return

    X_test = np.array([
        np.concatenate([r["tf_feat"], r["raw_feat"]])
        for r in last_day_records
    ], dtype=np.float32)

    scores = gbdt_model.predict(X_test)

    # 排序取 Top-K
    sorted_indices = np.argsort(scores)[::-1][:top_k]

    print(f"\n{'='*60}")
    print(f"LightGBM Stacking 预测 Top-{top_k} 股票 ({last_date}):")
    print(f"{'='*60}")
    selected = []
    for rank, idx in enumerate(sorted_indices):
        r = last_day_records[idx]
        print(f"  {rank+1}. {r['stock_code']}  score={scores[idx]:.4f}")
        selected.append((r["stock_code"], scores[idx]))

    # ─── 计算真实 5 日收益率 ──────────────────────
    test_df = pd.read_csv(test_csv_path)
    test_df["日期"] = pd.to_datetime(test_df["日期"])

    # 测试窗口: 5 个交易日，收益率 = (第5天开盘 - 第1天开盘) / 第1天开盘
    test_dates = sorted(test_df["日期"].unique())
    day1, day5 = test_dates[0], test_dates[-1]
    print(f"  测试窗口: {day1.date()} ~ {day5.date()}")

    day1_data = test_df[test_df["日期"] == day1].set_index("股票代码")
    day5_data = test_df[test_df["日期"] == day5].set_index("股票代码")

    returns = {}
    for stock in day1_data.index.intersection(day5_data.index):
        open_t1 = day1_data.loc[stock, "开盘"]
        open_t5 = day5_data.loc[stock, "开盘"]
        if open_t1 > 1e-4:
            returns[stock] = (open_t5 - open_t1) / open_t1

    # 计算组合加权收益（等权）
    total_return = 0.0
    valid_count = 0
    for stock_code, score in selected:
        if stock_code in returns:
            r = returns[stock_code]
            total_return += r
            valid_count += 1
            print(f"    {stock_code}: 5日真实收益率 = {r*100:.2f}%")
        else:
            print(f"    {stock_code}: ⚠️ 测试集中无数据")

    if valid_count > 0:
        avg_return = total_return / valid_count  # 等权
        print(f"\n{'='*60}")
        print(f"📊 等权组合 5 日收益率: {avg_return*100:.2f}%")
        print(f"   V7 (参考): 6.02% (等权) / 6.15% (收益门控)")
        print(f"   官方基线: 2.52%")
        print(f"   排行榜第1: 14.17%")
        print(f"{'='*60}")

    # 保存结果
    result_df = pd.DataFrame([
        {"排名": i+1, "股票代码": sc, "预测分数": score}
        for i, (sc, score) in enumerate(selected)
    ])
    result_path = os.path.join(OUTPUT_DIR, "gbdt_stacking_result.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8")
    print(f"\n结果已保存: {result_path}")

    return avg_return


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    # 1. 加载 V7 模型
    model, config, scaler_saved, stockid2idx, feature_columns = load_v7_model(device)

    # 2. 数据预处理
    print("\n--- 数据预处理 ---")
    if config.get("use_fundamentals", False):
        load_fundamentals()

    # 加载 & 划分数据
    from train import split_train_val_by_last_month, _preprocess_common, _build_label_and_clean
    data_file = os.path.join(DATA_DIR, "train.csv")
    full_df = pd.read_csv(data_file)
    train_df_raw, val_df_raw, val_start = split_train_val_by_last_month(
        full_df, config["sequence_length"], config.get("val_months", 3)
    )

    train_data, features = preprocess_data(train_df_raw, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df_raw, stockid2idx=stockid2idx)

    # ─── 测试集特殊处理：保留测试窗口日期（无 label）───
    test_df_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    # 用 train+test 拼接做特征工程，但需要保留无 label 的测试日期
    # 1) 先做特征工程（不过 _build_label_and_clean）
    combined_df = pd.concat([full_df, test_df_raw], ignore_index=True)
    combined_df = combined_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    feature_engineer = feature_engineer_func_map[config['feature_num']]

    groups = [group for _, group in combined_df.groupby('股票代码', sort=False)]
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(
            pool.imap(feature_engineer, groups),
            total=len(groups), desc="测试集特征工程"
        ))
    combined_processed = pd.concat(processed_list).reset_index(drop=True)
    combined_processed['instrument'] = combined_processed['股票代码'].map(stockid2idx)
    combined_processed = combined_processed.dropna(subset=['instrument']).copy()
    combined_processed['instrument'] = combined_processed['instrument'].astype(np.int64)

    # 标准化
    exclude_cols = ["股票代码", "日期", "datetime", "label", "label_abs",
                    "direction", "volatility"]
    feat_cols = [c for c in features if c not in exclude_cols and c in train_data.columns]

    train_data[feat_cols] = train_data[feat_cols].replace([np.inf, -np.inf], np.nan)
    val_data[feat_cols] = val_data[feat_cols].replace([np.inf, -np.inf], np.nan)
    combined_processed[feat_cols] = combined_processed[feat_cols].replace([np.inf, -np.inf], np.nan)

    train_data = train_data.dropna(subset=feat_cols)
    val_data = val_data.dropna(subset=feat_cols)
    combined_processed = combined_processed.dropna(subset=feat_cols)  # 不 drop label，保留 test 窗口

    local_scaler = StandardScaler()
    train_data[feat_cols] = local_scaler.fit_transform(train_data[feat_cols])
    val_data[feat_cols] = local_scaler.transform(val_data[feat_cols])
    combined_processed[feat_cols] = local_scaler.transform(combined_processed[feat_cols])
    combined_processed['label'] = 0.0  # 占位，test 日期无真实 label
    combined_processed['label_abs'] = 0.0

    # 合并 train+test — 取 test 窗口日期（2026-03-09 ~ 2026-03-13）
    test_dates_raw = sorted(pd.to_datetime(test_df_raw['日期'].unique()))
    print(f"特征列数: {len(feat_cols)}, 测试窗口: {test_dates_raw[0].date()} ~ {test_dates_raw[-1].date()}")

    print(f"训练集样本数: {len(train_data)}, 验证集样本数: {len(val_data)}")

    # 3. 提取 Transformer 特征
    print("\n--- 提取 Transformer 特征 (训练集) ---")
    train_records = extract_transformer_features(
        model, train_data, feat_cols, config["sequence_length"], stockid2idx, device
    )

    print("\n--- 提取 Transformer 特征 (验证集) ---")
    val_records = extract_transformer_features(
        model, val_data, feat_cols, config["sequence_length"], stockid2idx, device
    )

    print(f"\n--- 提取 Transformer 特征 (测试集: {test_dates_raw[-1].date()}) ---")
    test_records = extract_transformer_features(
        model, combined_processed, feat_cols, config["sequence_length"], stockid2idx, device
    )

    # 4. 构建 GBDT 数据集
    print("\n--- 构建 GBDT 数据集 ---")
    X_train, y_train, groups_train = build_gbdt_dataset(train_records)
    X_val, y_val, groups_val = build_gbdt_dataset(val_records)

    # 5. 训练 LightGBM
    gbdt_model = train_gbdt(X_train, y_train, groups_train, X_val, y_val, groups_val)

    # 保存模型
    gbdt_path = os.path.join(OUTPUT_DIR, "gbdt_stacking_model.txt")
    gbdt_model.save_model(gbdt_path)
    print(f"GBDT 模型已保存: {gbdt_path}")

    # 6. 测试集预测 & 评估
    print("\n--- 测试集评估 ---")
    test_csv_path = os.path.join(DATA_DIR, "test.csv")
    avg_return = predict_and_evaluate(gbdt_model, test_records, test_csv_path, top_k=5)

    # 7. 汇总对比
    print(f"\n{'='*60}")
    print(f"📊 最终对比:")
    print(f"   官方基线:              2.52%")
    print(f"   V6 (问题版):            0.46%")
    print(f"   V7 Transformer (等权):  6.02%")
    print(f"   V7 Transformer (门控):  6.15%")
    print(f"   GBDT Stacking (等权):   {avg_return*100:.2f}%" if avg_return else "   GBDT Stacking:          N/A")
    print(f"   排行榜第1名:           14.17%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

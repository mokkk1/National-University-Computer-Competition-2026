"""
walk_forward_predict.py — 使用 Walk-Forward 训练的所有模型做集成预测

预测策略:
  1. 加载所有窗口的所有 seed 模型
  2. 多模型 Median 分数聚合（P0）
  3. 多周期预测融合 — 平均最近 N 个预测日期的分数（P2）
  4. 一致性过滤 — 只保留多模型共识的股票（P5）
  5. Sharpe-like 权重分配

用法:
  python walk_forward_predict.py                           # 默认配置
  python walk_forward_predict.py --config light            # 轻量配置
  python walk_forward_predict.py --multi-period 5          # 5日预测融合
  python walk_forward_predict.py --top-k 5                 # Top-5
  python walk_forward_predict.py --output result.csv       # 指定输出
"""

import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import json
import argparse
from collections import Counter
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'src'))

from model import StockTransformer
from utils import engineer_features_158plus39, optimize_weights, select_top_stocks_with_gate
from train import feature_cloums_map, feature_engineer_func_map
from walk_forward import load_walk_forward_models, walk_forward_predict

# ─── 配置 ───────────────────────────────────────────────
DEFAULT_WF_DIR = os.path.join(os.path.dirname(__file__), 'model', 'walk_forward')
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__))
COMPETITION_DATA_DIR = os.environ.get(
    'CSI300_COMPETITION_DATA_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'data')
)


def compute_volatilities_from_raw(raw_close_df, stock_codes, pred_date):
    """从原始收盘价计算每只股票的近期波动率"""
    vols = []
    for code in stock_codes:
        sd = raw_close_df[
            (raw_close_df['股票代码'] == code) &
            (raw_close_df['日期'] <= pred_date)
        ].sort_values('日期')
        if len(sd) >= 20:
            closes = sd['收盘'].values.astype(float)
            rets = np.diff(closes) / (closes[:-1] + 1e-12)
            vols.append(np.std(rets[-20:]))
        else:
            vols.append(0.0)
    return np.array(vols)


def prepare_prediction_data(data_dir, scaler_for_features, config):
    """准备预测数据：加载+特征工程+标准化"""
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    print(f"\n📂 加载数据...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"   训练集: {train_df.shape[0]:,} 行, {train_df['日期'].min()} ~ {train_df['日期'].max()}")
    print(f"   测试集: {test_df.shape[0]:,} 行, {test_df['日期'].min()} ~ {test_df['日期'].max()}")

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    full_df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 特征工程
    feature_num = config.get('feature_num', '158+39')
    feature_engineer = feature_engineer_func_map.get(feature_num, engineer_features_158plus39)
    print(f"\n🔧 特征工程 ({feature_num})...")
    groups = [group for _, group in full_df.groupby('股票代码', sort=False)]
    processed_list = [feature_engineer(group) for group in tqdm(groups, desc="特征工程")]
    processed_df = pd.concat(processed_list).reset_index(drop=True)
    processed_df['日期'] = pd.to_datetime(processed_df['日期'])

    # 确定特征列
    all_feature_cols_template = feature_cloums_map.get(feature_num, feature_cloums_map['158+39'])
    feature_cols = [c for c in all_feature_cols_template if c in processed_df.columns]

    # 确保 instrument 列存在（与训练时一致，scaler 需要它）
    if 'instrument' not in processed_df.columns:
        all_stock_ids = processed_df['股票代码'].unique()
        stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
        processed_df['instrument'] = processed_df['股票代码'].map(stockid2idx)
        if 'instrument' not in feature_cols:
            feature_cols = ['instrument'] + feature_cols

    # 保存原始收盘价（标准化前！用于波动率计算）
    raw_close = processed_df[['股票代码', '日期', '收盘']].copy()

    # 获取一个 scaler 来推断特征维度（用第一个模型的 scaler）
    processed_df[feature_cols] = processed_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    processed_df[feature_cols] = processed_df[feature_cols].fillna(0)

    # 不在这里标准化，因为每个模型有自己的 scaler
    # 返回未标准化的数据，在预测时由每个模型独立处理

    return processed_df, feature_cols, raw_close, test_df


def multi_period_ensemble(models, processed_df, feature_cols, stock_codes,
                           test_dates, device, n_periods=5):
    """
    P2: 多周期预测融合。
    在最近的 N 个交易日上分别预测，然后聚合。

    策略：取每个日期预测分数的中位数。
    """
    if n_periods <= 1:
        pred_date = test_dates[-1] if len(test_dates) > 0 else processed_df['日期'].max()
        result = walk_forward_predict(
            models, processed_df, feature_cols,
            stock_codes, pred_date, device
        )
        if result is not None:
            result['预测日期'] = pred_date
        return result

    # 选择最近 N 个有数据的日期
    pred_dates = []
    for d in reversed(sorted(test_dates)):
        if len(pred_dates) >= n_periods:
            break
        pred_dates.append(d)
    pred_dates.reverse()

    print(f"\n📅 多周期融合: {len(pred_dates)} 个预测日期")
    for d in pred_dates:
        print(f"   {d}")

    all_scores = {}
    all_consistency = {}

    for pred_date in pred_dates:
        result = walk_forward_predict(
            models, processed_df, feature_cols,
            stock_codes, pred_date, device
        )
        if result is None:
            continue

        for _, row in result.iterrows():
            code = row['股票代码']
            if code not in all_scores:
                all_scores[code] = []
                all_consistency[code] = 0
            all_scores[code].append(row['预测分数'])
            all_consistency[code] = max(all_consistency[code], row.get('一致性', 0))

    # 聚合：取中位数
    agg_results = []
    for code in all_scores:
        agg_results.append({
            '股票代码': code,
            '预测分数': float(np.median(all_scores[code])),
            '分数标准差': float(np.std(all_scores[code])),
            '预测次数': len(all_scores[code]),
            '一致性': all_consistency[code],
        })

    result_df = pd.DataFrame(agg_results)
    result_df = result_df.sort_values('预测分数', ascending=False).reset_index(drop=True)
    return result_df


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward 集成预测')
    parser.add_argument('--config', type=str, default='light',
                        choices=['light', 'standard', 'v7'],
                        help='模型配置名称')
    parser.add_argument('--seeds', type=str, default='42',
                        help='随机种子，逗号分隔')
    parser.add_argument('--wf-dir', type=str, default=DEFAULT_WF_DIR,
                        help='Walk-Forward 模型目录')
    parser.add_argument('--data-dir', type=str, default=COMPETITION_DATA_DIR,
                        help='数据目录')
    parser.add_argument('--top-k', type=int, default=5, help='输出 Top-K')
    parser.add_argument('--multi-period', type=int, default=5,
                        help='多周期预测融合天数')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 CSV 路径')
    parser.add_argument('--consistency-threshold', type=float, default=0.4,
                        help='一致性过滤阈值 (0-1)')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 1. 加载所有模型
    print("=" * 60)
    print("Loading Walk-Forward Models...")
    print("=" * 60)
    models = load_walk_forward_models(args.wf_dir, args.config, seeds, device)
    print(f"加载了 {len(models)} 个模型")
    if len(models) == 0:
        print("❌ 没有找到模型！请先运行 walk_forward 训练。")
        sys.exit(1)

    for i, m in enumerate(models):
        print(f"  [{i+1}] seed={m['seed']}, dir={os.path.basename(m['window_dir'])}")

    # 2. 准备数据
    # 使用第一个模型的 scaler 推断特征维度
    first_cfg = models[0]['config']
    processed_df, feature_cols, raw_close, test_df = prepare_prediction_data(
        args.data_dir, models[0]['scaler'], first_cfg
    )

    # ⚠️ 标准化：需要使用各模型的 scaler
    # 为了简化，这里对所有模型使用各自的 scaler 在 predict 函数内部处理
    # 实际上 walk_forward_predict 使用的是未标准化的数据 — 需要修正

    # 但现在所有模型共享特征列定义，先标准化到第一个模型的 scaler 空间
    # 不同模型的 scaler 可能略有差异，但特征含义相同
    # 更好的做法是在 walk_forward_predict 中独立处理

    # 先用第一个模型的 scaler 标准化（作为默认）
    scaler = models[0]['scaler']
    # 只使用 scaler 知道的列（训练时拟合过的特征）
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = list(scaler.feature_names_in_)
    else:
        scaler_features = feature_cols  # fallback to feature_cols from prepare_prediction_data

    # 确保所有 scaler 期望的列都存在
    missing_from_scaler = [c for c in scaler_features if c not in processed_df.columns]
    if missing_from_scaler:
        print(f"⚠️ Scaler expects but missing: {missing_from_scaler}")
        for c in missing_from_scaler:
            processed_df[c] = 0.0  # 填充缺失列

    feature_cols_in_scaler = [c for c in scaler_features if c in processed_df.columns]
    processed_df[feature_cols_in_scaler] = processed_df[feature_cols_in_scaler].replace([np.inf, -np.inf], np.nan)
    processed_df[feature_cols_in_scaler] = processed_df[feature_cols_in_scaler].fillna(0)
    processed_df[feature_cols_in_scaler] = scaler.transform(processed_df[feature_cols_in_scaler])

    # 确定预测日期
    test_df['日期'] = pd.to_datetime(test_df['日期'])
    test_dates = sorted(test_df['日期'].unique())
    print(f"\n测试集日期范围: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} 天)")

    stock_codes = sorted(processed_df['股票代码'].unique())

    # 3. 多周期集成预测
    print("\n" + "=" * 60)
    print("Running Multi-Period Ensemble Prediction...")
    print("=" * 60)

    results = multi_period_ensemble(
        models, processed_df, feature_cols_in_scaler,
        stock_codes, test_dates, device,
        n_periods=args.multi_period
    )

    if results is None or len(results) == 0:
        print("❌ 预测失败！")
        sys.exit(1)

    # 4. 后处理
    top_k = args.top_k
    scores = results['预测分数'].values
    stock_codes_result = results['股票代码'].values

    # 计算波动率
    pred_date = pd.to_datetime(test_dates[-1])
    volatilities = compute_volatilities_from_raw(raw_close, stock_codes_result, pred_date)

    # ─── 一致性过滤（P5）───
    n_models = len(models)
    min_consensus = max(2, int(n_models * args.consistency_threshold))
    if '一致性' in results.columns:
        consistent_mask = results['一致性'] >= min_consensus
        n_consistent = consistent_mask.sum()
        print(f"\n一致性过滤: {n_consistent}/{len(results)} 只股票通过 "
              f"(≥{min_consensus}/{n_models} 模型共识)")

        if n_consistent >= top_k:
            # 只从一致性通过的股票中选
            consistent_idx = np.where(consistent_mask)[0]
            consistent_scores = scores[consistent_idx]
            consistent_vols = volatilities[consistent_idx]
            consistent_codes = stock_codes_result[consistent_idx]

            top_indices, top_weights = optimize_weights(
                consistent_scores, volatilities=consistent_vols,
                top_k=top_k, candidate_k=min(top_k * 2, len(consistent_scores)),
                use_volatility_penalty=True, temperature=2.0
            )
            # 映射回原始索引
            top_indices_orig = consistent_idx[top_indices]
        else:
            print(f"  ⚠️ 通过一致性的股票不足 {top_k} 只，使用全部股票")
            top_indices_orig, top_weights = optimize_weights(
                scores, volatilities=volatilities,
                top_k=top_k, candidate_k=min(top_k * 2, len(scores)),
                use_volatility_penalty=True, temperature=2.0
            )
    else:
        top_indices_orig, top_weights = optimize_weights(
            scores, volatilities=volatilities,
            top_k=top_k, candidate_k=min(top_k * 2, len(scores)),
            use_volatility_penalty=True, temperature=2.0
        )

    # 5. 输出结果
    print(f"\n{'='*60}")
    print(f"  📊 Walk-Forward Ensemble Top-{top_k} 选股结果")
    print(f"{'='*60}")

    top_results = []
    for i, (idx, w) in enumerate(zip(top_indices_orig, top_weights)):
        code = stock_codes_result[idx]
        entry = {
            '排名': i + 1,
            '股票代码': code,
            '预测分数': float(scores[idx]),
            '权重': float(w),
        }
        if '一致性' in results.columns:
            entry['一致性'] = int(results.iloc[idx]['一致性'])
        top_results.append(entry)
        print(f"  #{i+1}: {code}  score={scores[idx]:.4f}  weight={w:.4f}")

    top_df = pd.DataFrame(top_results)
    print(f"{'='*60}")

    # 保存
    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.wf_dir, f'ensemble_result_{ts}.csv')

    top_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果保存到: {output_path}")

    # 竞赛格式
    comp_format = top_df[['股票代码', '权重']].copy()
    comp_format.columns = ['stock_id', 'weight']
    comp_output = output_path.replace('.csv', '_comp.csv')
    comp_format.to_csv(comp_output, index=False, encoding='utf-8-sig')
    print(f"💾 竞赛格式保存到: {comp_output}")

    return top_df


if __name__ == '__main__':
    main()

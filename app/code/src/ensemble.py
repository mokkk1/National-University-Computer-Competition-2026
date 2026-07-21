"""
ensemble.py — 阶段4: 多模型集成推理

支持三种集成策略:
  1. mean: 简单平均所有模型的预测分数
  2. weighted: 按验证集 final_score 加权平均
  3. vote: 每个模型投票选出 Top-10，按得票数排序

用法:
    python ensemble.py                          # 使用默认配置集成
    python ensemble.py --mode vote              # 投票模式
    python ensemble.py --date 2026-03-06        # 指定预测日期
"""

import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import StockTransformer
from utils import engineer_features_158plus39, optimize_weights
from fundamental import (load_fundamentals, engineer_fundamental_features,
                         engineer_features_158plus39_fundamental, FUNDAMENTAL_FEATURE_COLS)
from train import feature_cloums_map


def find_model_dirs(base_dir):
    """自动发现所有已训练的模型目录"""
    candidates = [base_dir]
    # 也搜索 seed_* 变体
    parent = os.path.dirname(base_dir)
    if os.path.exists(parent):
        for d in os.listdir(parent):
            full = os.path.join(parent, d)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, 'best_model.pth')):
                if full not in candidates:
                    candidates.append(full)
    return [c for c in candidates if os.path.exists(os.path.join(c, 'best_model.pth'))]


def load_model(model_dir, device):
    """加载单个模型"""
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    input_dim = scaler.n_features_in_

    model = StockTransformer(input_dim=input_dim, config=config, num_stocks=300)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'best_model.pth'),
        map_location=device, weights_only=True
    ), strict=False)
    model.to(device)
    model.eval()

    # 获取 final_score 用于加权
    final_score = 0.0
    score_path = os.path.join(model_dir, 'final_score.txt')
    if os.path.exists(score_path):
        with open(score_path, 'r') as f:
            for line in f:
                if 'final_score' in line:
                    try:
                        final_score = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass

    return model, scaler, config, final_score


def load_models(model_dirs, device):
    """加载所有模型"""
    models = []
    for d in model_dirs:
        print(f"Loading model from: {d}")
        model, scaler, config, score = load_model(d, device)
        models.append({
            'model': model,
            'scaler': scaler,
            'config': config,
            'score': score,
            'dir': d
        })
    return models


def predict_ensemble(models_info, processed_df, feature_cols, stock_codes,
                     pred_date, pred_config, device, mode='weighted'):
    """
    集成预测。

    Args:
        models_info: load_models() 返回的模型列表
        processed_df: 预处理后的 DataFrame
        feature_cols: 特征列名列表
        stock_codes: 候选股票代码列表
        pred_date: 预测日期
        pred_config: 第一个模型的配置（取 sequence_length 等）
        device: 设备
        mode: 'mean' / 'weighted' / 'vote'

    Returns:
        pd.DataFrame with columns: 股票代码, 预测分数, 排名
    """
    seq_len = pred_config['sequence_length']
    all_scores = []  # [num_models, num_stocks]

    for mi in models_info:
        model = mi['model']
        config = mi['config']
        scaler = mi['scaler']

        # 为每个模型准备特征
        # (使用统一的特征列，因为 scaler 期望的特征数可能不同)
        model_features = feature_cols

        sequences = []
        valid_stocks = []
        for code in stock_codes:
            hist = processed_df[
                (processed_df['股票代码'] == code) &
                (processed_df['日期'] <= pred_date)
            ].sort_values('日期').tail(seq_len)

            if len(hist) == seq_len:
                # 只使用 scaler 能处理的特征
                available = [c for c in model_features if c in hist.columns]
                seq = hist[available].values.astype(np.float32)
                sequences.append(seq)
                valid_stocks.append(code)

        if len(sequences) == 0:
            continue

        seq_tensor = torch.FloatTensor(np.array(sequences)).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = model(seq_tensor).squeeze(0).cpu().numpy()
        all_scores.append((scores, valid_stocks, mi['score']))

    if not all_scores:
        return None

    # 统一股票列表（取交集）
    common_stocks = set(all_scores[0][1])
    for _, stocks, _ in all_scores[1:]:
        common_stocks &= set(stocks)
    common_stocks = sorted(common_stocks)

    # 创建统一的分数矩阵 [num_models, num_stocks]
    n_models = len(all_scores)
    score_matrix = np.zeros((n_models, len(common_stocks)))
    for i, (scores, stocks, _) in enumerate(all_scores):
        for j, code in enumerate(common_stocks):
            idx = stocks.index(code) if code in stocks else -1
            if idx >= 0:
                score_matrix[i, j] = scores[idx]

    # 集成策略
    if mode == 'mean':
        ensemble_scores = np.mean(score_matrix, axis=0)
    elif mode == 'weighted':
        weights = np.array([m['score'] for m in models_info])
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_models) / n_models
        ensemble_scores = np.average(score_matrix, axis=0, weights=weights)
    elif mode == 'vote':
        # 每个模型选出 top-10，按得票数排序
        votes = np.zeros(len(common_stocks))
        for i in range(n_models):
            top10_idx = np.argsort(score_matrix[i])[::-1][:10]
            votes[top10_idx] += 1
        ensemble_scores = votes
    else:
        raise ValueError(f"Unknown ensemble mode: {mode}")

    results = pd.DataFrame({
        '股票代码': common_stocks,
        '预测分数': ensemble_scores
    })
    results = results.sort_values('预测分数', ascending=False).reset_index(drop=True)
    results['排名'] = range(1, len(results) + 1)
    return results


def main():
    parser = argparse.ArgumentParser(description='Model Ensemble Inference')
    parser.add_argument('--mode', choices=['mean', 'weighted', 'vote'],
                        default='weighted', help='Ensemble mode')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Base model directory (default: <project>/model/)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory (default: <project root>)')
    parser.add_argument('--date', type=str, default=None,
                        help='Prediction date (YYYY-MM-DD)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Output Top-K stocks')
    args = parser.parse_args()

    # 项目根目录（自动推断）
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.model_dir is None:
        args.model_dir = os.path.join(_PROJECT_ROOT, 'model')
    if args.data_dir is None:
        args.data_dir = _PROJECT_ROOT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 查找模型
    model_dirs = find_model_dirs(args.model_dir)
    print(f"Found {len(model_dirs)} model(s):")
    for d in model_dirs:
        print(f"  {d}")

    if len(model_dirs) == 0:
        print("No trained models found!")
        sys.exit(1)

    models_info = load_models(model_dirs, device)

    # 加载数据
    print(f"\nLoading data...")
    train_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    full_df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # 特征工程（使用第一个模型的配置）
    base_config = models_info[0]['config']
    feature_num = base_config.get('feature_num', '158+39')

    print(f"Running feature engineering ({feature_num})...")
    groups = [g for _, g in full_df.groupby('股票代码', sort=False)]

    if 'fundamental' in feature_num:
        fundamentals_df = load_fundamentals()
        processed_list = []
        for g in tqdm(groups, desc="Feature engineering"):
            processed = engineer_features_158plus39_fundamental(g, fundamentals_df)
            processed_list.append(processed)
    else:
        processed_list = []
        for g in tqdm(groups, desc="Feature engineering"):
            processed = engineer_features_158plus39(g)
            processed_list.append(processed)

    processed_df = pd.concat(processed_list).reset_index(drop=True)
    processed_df['日期'] = pd.to_datetime(processed_df['日期'])

    all_stocks = sorted(processed_df['股票代码'].unique())
    stock2idx = {s: i for i, s in enumerate(all_stocks)}
    processed_df['instrument'] = processed_df['股票代码'].map(stock2idx).astype(np.int64)

    # 特征列
    base_features = feature_cloums_map[feature_num]
    feature_cols = [c for c in base_features if c in processed_df.columns]

    # 标准化（使用第一个模型的 scaler）
    scaler = models_info[0]['scaler']
    processed_df[feature_cols] = processed_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    processed_df[feature_cols] = scaler.transform(processed_df[feature_cols])

    # 预测
    pred_date = pd.to_datetime(args.date) if args.date else pd.to_datetime(test_df['日期'].max())
    print(f"\nPredicting at: {pred_date.date()} (mode: {args.mode})")

    results = predict_ensemble(
        models_info, processed_df, feature_cols,
        all_stocks, pred_date, base_config, device,
        mode=args.mode
    )

    if results is None:
        print("Prediction failed!")
        sys.exit(1)

    # 后处理
    volatilities = None
    if base_config.get('use_volatility_penalty', True):
        vols = []
        for code in all_stocks:
            sd = processed_df[processed_df['股票代码'] == code].sort_values('日期')
            if len(sd) >= 20:
                closes = sd['收盘'].values.astype(float)
                rets = np.diff(closes) / (closes[:-1] + 1e-12)
                vols.append(np.std(rets[-20:]))
            else:
                vols.append(0.0)
        volatilities = np.array(vols)

    top_indices, top_weights = optimize_weights(
        results['预测分数'].values,
        volatilities=volatilities,
        top_k=args.top_k,
        candidate_k=base_config.get('post_top_k', 10),
        use_volatility_penalty=base_config.get('use_volatility_penalty', True),
        temperature=2.0
    )

    top_results = results.iloc[top_indices].copy()
    top_results['权重'] = top_weights
    top_results['排名'] = range(1, len(top_results) + 1)

    print(f"\n{'='*60}")
    print(f"  Ensemble Top-{args.top_k} ({args.mode} mode)")
    print(f"{'='*60}")
    for _, row in top_results.iterrows():
        print(f"  {row['股票代码']}  score={row['预测分数']:.4f}  weight={row['权重']:.4f}")
    print(f"{'='*60}")

    # 保存
    output_dir = os.path.join(args.model_dir, '..')
    output_path = os.path.join(output_dir, f'ensemble_{args.mode}_{pred_date.strftime("%Y%m%d")}.csv')
    top_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to: {output_path}")

    # 竞赛格式
    comp_format = top_results[['股票代码', '权重']].copy()
    comp_format.columns = ['stock_id', 'weight']
    comp_output = os.path.join(output_dir, f'result_ensemble_{pred_date.strftime("%Y%m%d")}.csv')
    comp_format.to_csv(comp_output, index=False, encoding='utf-8-sig')
    print(f"Competition format saved to: {comp_output}")

    return top_results


if __name__ == '__main__':
    main()

"""
test.py — 生成 Top-5 选股预测结果 (Docker版本)
用法: python test.py

输出: /app/output/result.csv (stock_id, weight 格式)
"""
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import json
import random
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import optimize_weights
from fundamental import (load_fundamentals, engineer_features_158plus39_fundamental,
                         FUNDAMENTAL_FEATURE_COLS)
from train import feature_cloums_map


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_model_and_scaler(model_dir, device):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        saved_config = json.load(f)

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    scaler = joblib.load(scaler_path)

    model_path = os.path.join(model_dir, 'best_model.pth')
    input_dim = scaler.n_features_in_

    model = StockTransformer(input_dim=input_dim, config=saved_config, num_stocks=300)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print(f"Model loaded: {model_path}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Input dim: {input_dim}")
    print(f"  Device: {device}")
    return model, scaler, saved_config


def load_and_preprocess_data(data_dir, scaler, saved_config):
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    print(f"Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  Train: {train_df.shape[0]:,} rows")
    print(f"  Test:  {test_df.shape[0]:,} rows")

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    full_df = full_df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    # Feature engineering
    feature_num = saved_config.get('feature_num', '158+39')
    print(f"Running feature engineering ({feature_num})...")

    groups = [g for _, g in full_df.groupby('股票代码', sort=False)]

    if 'fundamental' in feature_num:
        fundamentals_df = load_fundamentals()
        processed_list = []
        for g in tqdm(groups, desc="Feature engineering"):
            processed = engineer_features_158plus39_fundamental(g, fundamentals_df)
            processed_list.append(processed)
    else:
        from utils import engineer_features_158plus39
        processed_list = []
        for g in tqdm(groups, desc="Feature engineering"):
            processed = engineer_features_158plus39(g)
            processed_list.append(processed)

    processed_df = pd.concat(processed_list).reset_index(drop=True)
    processed_df['日期'] = pd.to_datetime(processed_df['日期'])

    # Stock index mapping
    all_stock_ids = processed_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    processed_df['instrument'] = processed_df['股票代码'].map(stockid2idx).astype(np.int64)

    # Feature columns matching scaler
    expected = list(scaler.feature_names_in_)
    feature_cols = [c for c in expected if c in processed_df.columns]
    missing = set(expected) - set(feature_cols)
    if missing:
        print(f"Warning: missing features ({len(missing)}): {list(missing)[:5]}...")

    print(f"  Using {len(feature_cols)} features (scaler expects {len(expected)})")

    # Normalize
    print(f"Normalizing...")
    processed_df[feature_cols] = processed_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    processed_df[feature_cols] = processed_df[feature_cols].fillna(0)
    processed_df[feature_cols] = scaler.transform(processed_df[feature_cols])

    return processed_df, feature_cols


def predict_for_date(data, features, stock_codes, date, model, saved_config, device):
    """Predict ranking scores for all stocks on a given date"""
    sequence_length = saved_config['sequence_length']

    sequences = []
    valid_stocks = []

    for code in stock_codes:
        stock_hist = data[
            (data['股票代码'] == code) &
            (data['日期'] <= date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_hist) == sequence_length:
            seq = stock_hist[features].values.astype(np.float32)
            sequences.append(seq)
            valid_stocks.append(code)

    if len(sequences) == 0:
        print(f"Error: no data for date {date}")
        return None

    seq_tensor = torch.FloatTensor(np.array(sequences)).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(seq_tensor).squeeze(0).cpu().numpy()

    results = pd.DataFrame({
        'stock_id': valid_stocks,
        'score': scores
    })
    results = results.sort_values('score', ascending=False).reset_index(drop=True)
    results['rank'] = range(1, len(results) + 1)
    return results


def main():
    set_seed(42)

    # Docker paths — 优先从 config 读取（训练和推理共用同一目录），
    # 其次用环境变量覆盖，最后回退到默认值
    BASE_DIR = os.environ.get('CSI300_BASE_DIR', '/app')
    DATA_DIR = os.environ.get('CSI300_DATA_DIR',
                              os.path.join(BASE_DIR, 'data'))
    OUTPUT_DIR = os.environ.get('CSI300_OUTPUT_DIR',
                                os.path.join(BASE_DIR, 'output'))
    # 模型目录：优先环境变量，其次从 config 读取（与 train.py 保持一致），最后回退
    if 'CSI300_MODEL_DIR' in os.environ:
        MODEL_DIR = os.environ['CSI300_MODEL_DIR']
    else:
        MODEL_DIR = config.get('output_dir',
                               os.path.join(BASE_DIR, 'model',
                                            '60_158+39+fundamental+momentum_v8_improved'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # 1. Load model
    model, scaler, saved_config = load_model_and_scaler(MODEL_DIR, device)

    # 2. Load and preprocess data
    processed_df, feature_cols = load_and_preprocess_data(DATA_DIR, scaler, saved_config)

    # 3. Determine prediction date (latest in test.csv)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    test_df['日期'] = pd.to_datetime(test_df['日期'])
    pred_date = test_df['日期'].max()
    print(f"\nPrediction date: {pred_date.strftime('%Y-%m-%d')}")

    # 4. Predict
    stock_codes = sorted(processed_df['股票代码'].unique())
    results = predict_for_date(processed_df, feature_cols, stock_codes,
                               pred_date, model, saved_config, device)

    if results is None:
        sys.exit(1)

    # 5. Compute volatilities for risk adjustment
    volatilities = None
    if saved_config.get('use_volatility_penalty', True):
        vols = []
        for code in stock_codes:
            sd = processed_df[processed_df['股票代码'] == code].sort_values('日期')
            if len(sd) >= 20:
                closes = sd['收盘'].values.astype(float)
                rets = np.diff(closes) / (closes[:-1] + 1e-12)
                vols.append(np.std(rets[-20:]))
            else:
                vols.append(0.0)
        volatilities = np.array(vols)

    # 6. Post-process with optimized weights
    top_indices, top_weights = optimize_weights(
        results['score'].values,
        volatilities=volatilities,
        top_k=5,
        candidate_k=saved_config.get('post_top_k', 10),
        use_volatility_penalty=saved_config.get('use_volatility_penalty', True),
        temperature=2.0
    )

    top5 = results.iloc[top_indices].copy()
    top5['weight'] = top_weights
    top5 = top5[['stock_id', 'weight']]

    print(f"\n{'='*50}")
    print(f"  Top-5 Prediction — {pred_date.strftime('%Y-%m-%d')}")
    print(f"{'='*50}")
    for _, row in top5.iterrows():
        print(f"  {row['stock_id']}  {row['weight']:.4f}")
    print(f"  Weight sum: {top5['weight'].sum():.4f}")
    print(f"{'='*50}")

    # 7. Save
    output_path = os.path.join(OUTPUT_DIR, 'result.csv')
    top5.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_path}")

    return top5


if __name__ == '__main__':
    main()

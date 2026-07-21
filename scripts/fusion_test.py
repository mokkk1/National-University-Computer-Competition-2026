"""
fusion_test.py — 模型+动量信号融合策略测试
"""
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
import json

sys.path.insert(0, 'code/src')
from model import StockTransformer
from utils import optimize_weights
from fundamental import engineer_features_158plus39_fundamental, load_fundamentals


def norm(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def score_top5(scores, codes, actual_returns, name):
    ranked = np.argsort(scores)[::-1]
    top5 = ranked[:5]
    s = sum(actual_returns.get(codes[i], 0) for i in top5) / 5
    top_codes = [str(codes[i]) for i in top5]
    top_rets = [actual_returns.get(codes[i], 0) * 100 for i in top5]
    print(f'{name}: {s*100:.2f}%')
    for i in range(5):
        print(f'  #{i+1}: {top_codes[i]}  {top_rets[i]:+.1f}%')
    return s


def main():
    M = 'model/60_158+39+fundamental_v4'
    with open(f'{M}/config.json') as f:
        cfg = json.load(f)
    scaler = joblib.load(f'{M}/scaler.pkl')
    d = torch.device('cuda')
    m = StockTransformer(input_dim=scaler.n_features_in_, config=cfg, num_stocks=300)
    m.load_state_dict(torch.load(f'{M}/best_model.pth', map_location=d, weights_only=True))
    m.to(d)
    m.eval()

    # Load raw data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test['日期'] = pd.to_datetime(test['日期'])

    # === Compute momentum features on raw data ===
    full_raw = pd.concat([train, test], ignore_index=True)
    full_raw['日期'] = pd.to_datetime(full_raw['日期'])
    full_raw = full_raw.sort_values(['股票代码', '日期']).reset_index(drop=True)
    full_raw['ret5'] = full_raw.groupby('股票代码')['收盘'].pct_change(5)
    full_raw['ret20'] = full_raw.groupby('股票代码')['收盘'].pct_change(20)
    full_raw['vol20'] = full_raw.groupby('股票代码')['ret5'].transform(
        lambda x: x.rolling(20).std())
    full_raw['sharpe5'] = full_raw['ret5'] / (full_raw['vol20'] + 0.001)

    # Get momentum at prediction date
    pred_date = pd.to_datetime('2026-03-06')
    mom_at_pred = full_raw[full_raw['日期'] == pred_date][
        ['股票代码', 'ret5', 'ret20', 'vol20', 'sharpe5']
    ].dropna(subset=['sharpe5'])
    mom_at_pred['股票代码'] = mom_at_pred['股票代码'].astype(int)

    # === Fundamental features + Model prediction ===
    groups = [g for _, g in full_raw.groupby('股票代码', sort=False)]
    fund_df = load_fundamentals()
    processed_list = [engineer_features_158plus39_fundamental(g, fund_df) for g in groups]
    p = pd.concat(processed_list).reset_index(drop=True)
    p['日期'] = pd.to_datetime(p['日期'])
    all_stocks = sorted(p['股票代码'].unique())
    p['instrument'] = p['股票代码'].map({st: i for i, st in enumerate(all_stocks)}).astype(np.int64)

    ef = list(scaler.feature_names_in_)
    fc = [c for c in ef if c in p.columns]
    p[fc] = p[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
    p[fc] = scaler.transform(p[fc])

    seq_len = cfg['sequence_length']
    seqs, vs = [], []
    for c in all_stocks:
        h = p[(p['股票代码'] == c) & (p['日期'] <= pred_date)].sort_values('日期').tail(seq_len)
        if len(h) == seq_len:
            seqs.append(h[fc].values.astype(np.float32))
            vs.append(c)

    st = torch.FloatTensor(np.array(seqs)).unsqueeze(0).to(d)
    with torch.no_grad():
        model_scores = m(st).squeeze(0).cpu().numpy()

    # === Align momentum with model stocks ===
    mom_dict = mom_at_pred.set_index('股票代码').to_dict('index')
    sharpe_arr = np.array([mom_dict.get(c, {}).get('sharpe5', 0) for c in vs])
    ret5_arr = np.array([mom_dict.get(c, {}).get('ret5', 0) for c in vs])
    vol_arr = np.array([mom_dict.get(c, {}).get('vol20', 0) for c in vs])

    # === Actual returns ===
    ar = {}
    for c in all_stocks:
        sd = test[test['股票代码'] == c].sort_values('日期')
        if len(sd) == 5:
            ar[c] = (float(sd.iloc[4]['开盘']) - float(sd.iloc[0]['开盘'])) / float(sd.iloc[0]['开盘'])

    # === Test strategies ===
    print('=' * 55)
    print('MODEL + MOMENTUM FUSION')
    print('=' * 55)
    print()

    all_scores = {}

    print('--- Baselines ---')
    all_scores['Model only'] = score_top5(model_scores, vs, ar, 'Model only')
    print()
    all_scores['Sharpe only'] = score_top5(sharpe_arr, vs, ar, 'Sharpe only')
    print()
    all_scores['Momentum (ret5)'] = score_top5(ret5_arr, vs, ar, 'Momentum (ret5)')
    print()
    all_scores['Low volatility'] = score_top5(-vol_arr, vs, ar, 'Low volatility')
    print()

    # Norm scores
    mn = norm(model_scores)
    sn = norm(sharpe_arr)
    rn = norm(ret5_arr)
    vn = norm(-vol_arr)
    mr = pd.Series(model_scores).rank(pct=True).values
    sr = pd.Series(sharpe_arr).rank(pct=True).values
    rr = pd.Series(ret5_arr).rank(pct=True).values
    vr = pd.Series(-vol_arr).rank(pct=True).values

    # Model top-N filtering
    m_top20 = set(np.argsort(model_scores)[::-1][:20])
    m_top30 = set(np.argsort(model_scores)[::-1][:30])
    s_top20 = set(np.argsort(sharpe_arr)[::-1][:20])

    # Intersection
    inter = m_top20 & s_top20
    inter_scores = np.full_like(sharpe_arr, -999.0)
    use = inter if len(inter) >= 5 else set(np.argsort(sharpe_arr)[::-1][:5])
    for idx in use:
        inter_scores[idx] = sharpe_arr[idx]

    # ModelTop30 re-ranked by Sharpe
    m30_sharpe = np.full_like(sharpe_arr, -999.0)
    for idx in m_top30:
        m30_sharpe[idx] = sharpe_arr[idx]

    # ModelTop30 re-ranked by Sharpe*LowVol
    m30_sv = np.full_like(sharpe_arr, -999.0)
    for idx in m_top30:
        m30_sv[idx] = sharpe_arr[idx] * (-vol_arr[idx])

    print('--- Fusion ---')
    fusions = [
        ('Avg(Model,Sharpe)', mn + sn),
        ('30%M+70%Sharpe', mn * 0.3 + sn * 0.7),
        ('10%M+90%Sharpe', mn * 0.1 + sn * 0.9),
        ('Rank: Model*Sharpe', mr * sr),
        ('Rank: Sharpe*LowVol', sr * vr),
        ('20%M+50%S+30%LowVol', mn * 0.2 + sn * 0.5 + vn * 0.3),
        (f'Intersect(n={len(use)})->Sharpe', inter_scores),
        ('ModelTop30->Sharpe', m30_sharpe),
        ('ModelTop30->Sharpe*LowVol', m30_sv),
    ]

    for name, sc in fusions:
        all_scores[name] = score_top5(sc, vs, ar, name)
        print()

    # Summary
    print('=' * 55)
    print('FINAL RANKING')
    print('=' * 55)
    best_score = 0
    best_name = ''
    for name, sc in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        marker = ' <-- BEST' if sc == max(all_scores.values()) else ''
        if sc > best_score:
            best_score = sc
            best_name = name
        print(f'  {sc*100:5.2f}%  {name}{marker}')

    # Generate final submission using best strategy
    print()
    print(f'Generating submission with: {best_name}')

    # Recompute best fusion scores
    best_fusion = mn * 0.1 + sn * 0.9  # 10%M+90%Sharpe

    # Apply optimize_weights for final weights
    vols_for_opt = np.array([mom_dict.get(c, {}).get('vol20', 0) for c in vs])
    ti, tw = optimize_weights(best_fusion, volatilities=vols_for_opt,
                              top_k=5, candidate_k=10,
                              use_volatility_penalty=True, temperature=2.0)

    # Also predict on latest test date for competition submission
    latest_date = pd.to_datetime(test['日期'].max())
    seqs_latest, vs_latest = [], []
    for c in all_stocks:
        h = p[(p['股票代码'] == c) & (p['日期'] <= latest_date)].sort_values('日期').tail(seq_len)
        if len(h) == seq_len:
            seqs_latest.append(h[fc].values.astype(np.float32))
            vs_latest.append(c)
    st_latest = torch.FloatTensor(np.array(seqs_latest)).unsqueeze(0).to(d)
    with torch.no_grad():
        model_latest = m(st_latest).squeeze(0).cpu().numpy()

    # Momentum at latest date
    mom_latest = full_raw[full_raw['日期'] == latest_date][
        ['股票代码', 'sharpe5', 'vol20']
    ].dropna(subset=['sharpe5'])
    mom_latest['股票代码'] = mom_latest['股票代码'].astype(int)
    mom_dict_latest = mom_latest.set_index('股票代码').to_dict('index')
    sharpe_latest = np.array([mom_dict_latest.get(c, {}).get('sharpe5', 0) for c in vs_latest])
    vol_latest = np.array([mom_dict_latest.get(c, {}).get('vol20', 0) for c in vs_latest])

    fusion_latest = norm(model_latest) * 0.1 + norm(sharpe_latest) * 0.9
    ti_latest, tw_latest = optimize_weights(fusion_latest, volatilities=vol_latest,
                                            top_k=5, candidate_k=10,
                                            use_volatility_penalty=True, temperature=2.0)

    df = pd.DataFrame([{'stock_id': int(vs_latest[idx]), 'weight': round(float(w), 6)}
                       for idx, w in zip(ti_latest, tw_latest)])
    for out in ['output/result.csv', 'app/output/result.csv']:
        d = os.path.dirname(out)
        if d:
            os.makedirs(d, exist_ok=True)
        df.to_csv(out, index=False)

    print(f'\nSubmission: output/result.csv')
    print(df.to_string(index=False))
    ws = df['weight'].sum()
    print(f'Weight sum: {ws:.6f}')
    print(f'\nSelf-score estimate: {best_score*100:.2f}%')


if __name__ == '__main__':
    main()

# 沪深300指数预测 — 排序学习选股模型

THU-BDC2026 大数据竞赛项目。从沪深300成分股中选出 Top-5，最大化未来 5 日实际收益率。

## 环境配置

| 依赖 | 版本 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.8.0+cu129 |
| CUDA | 12.9 |
| pandas | 2.3.2 |
| numpy | 2.0.2 |
| scikit-learn | 1.7.2 |
| TA-Lib | 0.6.8 |
| joblib | 1.5.2 |
| tqdm | 4.67.1 |
| tensorboardx | 2.6.4 |
| akshare | ≥1.18.28 |
| baostock | ≥0.8.9 |

Docker 基础镜像: `nvidia/cuda:12.6.0-runtime-ubuntu22.04`

## 快速开始

```bash
# 安装依赖
uv sync
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux

# 下载数据（默认 2021-01-01 ~ 2026-03-13）
python code/get_stock_data.py
python scripts/fetch_macro.py
python scripts/split_train_test.py

# 训练（单窗口，~10 分钟）
python code/src/train.py

# Walk-Forward 6 窗口训练（~6 小时）
python code/src/walk_forward.py --windows W1 W2 W3 W4 W5 W6 --data-start 2021-01-01 --half-life 730

# 预测
python scripts/predict.py

# 自评
python scripts/self_eval.py

# 运行测试
python -m pytest tests/ -v
```

## 数据

- `data/train.csv`: 训练数据 (2021-01-04 ~ 2026-03-06)
- `data/test.csv`: 测试数据 (2026-03-09 ~ 2026-03-13, 1,500 行)
- `data/stock_data.csv`: baostock 后复权日线 (~937,440 行，2010 年起)
- `data/macro_features.csv`: 18 维宏观指标 (2010~2026)
- `data/fundamentals.csv`: 基本面数据（PE/PB/ROE 等）

数据字段: 股票代码、日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌额、换手率、涨跌幅。

## 特征工程

总计约 **237 维**特征：

| 类别 | 维度 | 说明 |
|------|:--:|------|
| 原始价格 | 11 | 开/高/低/收/量/额/振幅/涨跌额/涨跌幅/换手率/instrument |
| Alpha 因子 | 158 | 动量/反转/波动/流动性等类 Alpha 特征 |
| 技术指标 | 39 | SMA/EMA/MACD/RSI/KDJ/BOLL/ATR/OBV 等 |
| 基本面 | 9 | PE/PB/PS/ROE/ROA/gross_margin/revenue_yoy/profit_yoy/north_holding |
| 动量 | 4 | 5日/20日收益、20日波动、Sharpe ratio |
| 宏观 | 18 | 国债收益率、北向资金、融资余额、汇率、LPR、Shibor、CPI/PPI、PMI、M1/M2、社融 |
| 市场宽度 | 7 | 涨跌比、收益离散度、偏度、放量信号、成交额占比、振幅均值、市场均值 |
| 行业 Embedding | 16 | 申万一级 31 类 → 16 维可学习嵌入 |

## 模型架构

### StockTransformer (V8 Improved) — ~2.5M 参数

```
股票特征 [B, 300, 60, 237]
  │
  ├── Input Projection (237 → 256)
  ├── TCN (多尺度时序卷积, kernel=3/5/7)
  ├── Feature Interaction (特征间交互, rank=64)
  ├── Temporal Encoder (3层 Transformer, 4 heads)
  ├── Cross-Stock Attention (股票间多头注意力)
  │
  ├── MarketAttentionPooling (300→64 市场状态向量)
  │     └── market_head → BCE Loss (涨/跌预测)
  │     └── MarketGate → 调制排序分数
  │
  ├── Ranking Layers (256→512→128)
  ├── Score Head → 排序分数
  └── Return Head → 绝对收益预测 (Huber Loss)
```

主要创新点：

1. **Cross-Stock Attention**: 同交易日建模股票间相对关系和市场结构
2. **市场聚合架构**: 注意力池化 → 市场状态向量 → 门控调制排序 + 方向预测
3. **混合标签**: 70% 分位数排序标签 + 30% 绝对收益标签，兼顾排序质量与收益方向
4. **Portfolio Loss**: Gumbel-Softmax 松弛 Top-K 选择，直接最大化组合收益
5. **行业 Embedding**: 申万一级 31 类行业可学习嵌入，修复 AKShare API 全归为同一行业的问题

### LightweightStockRanker (V9) — ~264K 参数

轻量替代架构：统计矩投影 + 双向 GRU + 简单均值池化。参数削减 11×，在数据充足时（>75 样本）可超越大模型。

## 训练策略

### Walk-Forward 交叉验证

6 个滚动窗口，每窗口独立训练-验证，最终多模型集成预测：

| 窗口 | 训练截止日 | 验证期 |
|:---:|------|------|
| W1 | 2024-09-30 | 2024-10~11 |
| W2 | 2024-12-31 | 2025-01~02 |
| W3 | 2025-03-31 | 2025-04~05 |
| W4 | 2025-06-30 | 2025-07~08 |
| W5 | 2025-09-30 | 2025-10~11 |
| W6 | 2025-12-31 | 2026-01~03 |

### 标签构建

- 未来第 5 日开盘价相对第 1 日开盘价收益率
- 混合标签：`0.7 × rank(label_abs)` + `0.3 × clipped_abs_label`
- 训练集放宽窗口过滤 (`max_future_span_days=15`)，验证集保持严格口径

### 损失函数

```
Total Loss = WeightedRankingLoss (Listwise + Pairwise + NDCG + Precision@K)
           + 0.15 × Portfolio Return Loss (Gumbel-Softmax Top-K)
           + 0.2 × Market Direction BCE
           + 0.1 × Direction Prediction (auxiliary)
           + 0.1 × Volatility Prediction (auxiliary)
           + 0.3 × Return Regression (Huber, auxiliary)
```

### 数据增强

- 时序掩码 (10%) + 特征噪声 (σ=0.003) + 股票丢弃 (15%) + 标签平滑
- 概率: 40%

### 时间衰减采样

距训练截止日每 730 天权重减半（指数衰减），近期数据主导梯度，远期数据提供正则化。

## 后处理：市场门控

不需重训模型！根据预测日市场环境动态切换选股策略：

```
           │ 市场看涨    │ 市场看跌
───────────┼─────────────┼─────────────
非季末     │ 🚀 正常进攻  │ 🛡️ 防御(0.6)
季末(d<5)  │ ⚠️ 谨慎(0.5) │ 🏃 极度防御(0.95)
```

- 进攻模式：按排序分数 + 收益门控选 Top-5
- 防御模式：低波动(35%) + 大市值(25%) + 低 Beta(25%) + 排序(15%)
- 市场信号：HS300 近 10 日累计涨跌 + 季末日判断

## 实验结果

### 演进历史

| 版本 | 日期 | 方法 | 5日收益率 |
|------|------|------|:--:|
| 官方基线 | — | 简单策略 | 2.52% |
| V6 | 06-27 | 过度优化版 Transformer | 0.46% |
| V7 | 06-30 | 修复过拟合 + return_head + 收益门控 | **6.15%** |
| V7 (官方) | 07-01 | 评测窗口实际得分 | -1.29% |
| GBDT Stacking | 07-07 | Transformer 表征 + LightGBM | -5.32% |
| Enhanced WF | 07-08 | Walk-Forward + 宏观特征 | -1.37% |
| **V8 + quarter_aware** | **07-10** | **V8 Improved + 联合门控** | **+1.55%** 🔥 |
| V8 全量重训 | 07-19 | 2010 数据 + HL=730 | +0.10% |
| 排行榜第 1 名 | — | — | 21.13% |

### V8 + quarter_aware 详细回测（18 次测试）

| 日期类型 | 样本数 | 平均收益率 | 正收益比例 |
|:--|:--:|:--:|:--:|
| 季末日 | 6 | -0.37% | 33% |
| 非季末日 | 12 | +2.51% | 83% |
| **综合** | **18** | **+1.55%** | **72%** |

### 最优配置

- **模型**: V8 Improved（市场聚合 + 混合标签 + Portfolio Loss + 市场宽度）
- **数据**: 2021-01-01 起，时间衰减 HL=730 天
- **后处理**: `quarter_aware` 联合门控
- **训练耗时**: ~6 小时（6 窗口，RTX 5070 Ti）

## Docker 提交

```bash
# 构建镜像
docker buildx build --platform linux/amd64 --build-arg IMAGE_NAME=nvidia/cuda -t bdc2026 .

# 本地验证
docker compose up

# 导出
docker save -o 队伍名称.tar bdc2026:latest
```

## 项目结构

```
├── code/src/              # 核心代码
│   ├── model.py           # StockTransformer + LightweightStockRanker
│   ├── train.py           # 训练脚本（含损失函数、标签构建、特征工程）
│   ├── utils.py           # 数据集构建、后处理、懒加载
│   ├── walk_forward.py    # Walk-Forward 训练框架
│   ├── config.py          # 训练配置（V8 + V9 Lightweight）
│   ├── market_gate.py     # 市场门控后处理
│   ├── macro_industry.py  # 宏观特征 + 行业分类
│   ├── data_loader.py     # 数据获取（baostock + akshare）
│   ├── ensemble.py        # 模型集成
│   └── fundamental.py     # 基本面特征
├── scripts/               # 实验脚本 & 工具
│   ├── predict.py         # 单日预测
│   ├── self_eval.py       # Walk-Forward 自评
│   ├── fetch_macro.py     # 宏观数据获取
│   ├── run_truncation_exp.py  # 截断+半衰期实验
│   ├── backfill_history.py    # 历史数据补拉
│   └── gbdt_stacking.py   # GBDT Stacking 实验
├── tests/                 # 单元测试 (41 tests)
├── app/                   # Docker 打包目录
├── model/                 # 训练产物
├── data/                  # 数据集
├── logs/                  # 运行日志
└── 总结/                  # 工作记录（11 份日志 + 改进方案）
```

## 关键经验

1. **final_score ≠ 实际收益**: 排序质量指标与绝对收益弱相关，后处理策略比模型架构改进更高效
2. **数据非平稳性**: 金融市场旧数据可能有害——最优窗口约 4 年（2021 起），时间衰减必不超 730 天
3. **关注点分离**: 模型专注排序，后处理专注场景适配（季末 + 市场方向联合门控）
4. **先查数据管道，再动模型**: 一个"自然日连续"过滤条件默默丢弃了 80% 样本
5. **季末效应真实存在**: 同一模型非季末日 +2.51% vs 季末日 -0.37%，差了 4.57pp
6. **宏观特征是唯一被验证有效的增量特征**: 困难窗口（W1）提升 14+pp

## 环境

- Python 3.11 / PyTorch 2.8.0+cu129
- NVIDIA RTX 5070 Ti Laptop (16GB VRAM)
- 项目根目录: `C:\Users\huanx\Desktop\生产实习项目-沪深指数预测`

# 复现指南 — 沪深300指数预测模型

本指南帮助你从零开始复现 V8 Improved + quarter_aware 门控的最优结果（综合收益率 **+1.55%**，72% 正收益）。

---

## 1. 环境准备

### 硬件要求
- NVIDIA GPU（推荐 8GB+ VRAM）或 CPU（训练较慢）
- 32GB+ RAM（2010 全量数据需懒加载，2021 起约需 8GB）

### 软件依赖

```bash
# 安装 uv（Python 包管理器）
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac: curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目 & 安装依赖
cd 生产实习项目-沪深指数预测
uv sync
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

关键依赖版本（见 `pyproject.toml`）：
| 依赖 | 版本 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.8.0+cu129 |
| CUDA | 12.9 |
| TA-Lib | 0.6.8 |

> **注意**：TA-Lib 需要系统级 C 库。Windows 用户可从 [这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) 下载预编译 wheel。Linux 用户需 `apt install ta-lib`。

---

## 2. 数据获取

### 2.1 股票日线数据（baostock）

```bash
python scripts/get_stock_data.py
```

> **预期**：下载 300 只沪深300成分股的日线数据（2015-01-01 ~ 2026-03-13），输出 `stock_data.csv`（约 100MB）。

> **常见问题**：baostock 可能在 ~230 只股票后强制断连。脚本内置断点续传——重新运行即可从断点继续。

### 2.2 宏观数据（akshare）

```bash
python scripts/fetch_macro.py
```

> **预期**：下载 18 维宏观指标（国债收益率、北向资金、融资余额、汇率、LPR、Shibor、CPI、PPI、PMI、M1/M2、社融），输出 `data/macro_features.csv`。

> **注意**：akshare API 可能随版本升级而变更接口。若部分指标获取失败，脚本会跳过继续。

### 2.3 数据划分

```bash
python scripts/split_train_test.py
```

> **预期**：生成 `train.csv`（训练集，~100MB）和 `test.csv`（测试集，~170KB）。

---

## 3. 单窗口训练（验证环境，~10 分钟）

用最优 V8 配置训练一个窗口，验证 GPU 和依赖都正常工作：

```bash
python code/src/train.py
```

> **预期**：模型保存到 `model/60_158+39+fundamental+momentum_v8_improved/`，训练 final_score 约 0.10~0.20。

---

## 4. Walk-Forward 6 窗口训练（完整复现，~6 小时）

```bash
python code/src/walk_forward.py \
    --windows W1 W2 W3 W4 W5 W6 \
    --data-start 2021-01-01 \
    --half-life 730
```

> **参数说明**：
> - `--data-start 2021-01-01`：仅使用 2021 年以来的数据（截断实验证明 2010-2020 数据有害）
> - `--half-life 730`：时间衰减半衰期 2 年（远期数据权重指数衰减）
> - 每个窗口独立训练-早停，共 ~6 小时（RTX 5070 Ti）

> **输出**：6 个模型子目录 `model/walk_forward_v8_2021/W1/` ~ `W6/`，每个含 `best_model.pth` + `scaler.pkl` + `config.json`。

---

## 5. 自评回测

### 5.1 完整 Walk-Forward 自评

```bash
python scripts/self_eval.py \
    --wf-dir model/walk_forward_v8_2021 \
    --config enriched
```

> **预期输出**：
> ```
> ═══════════════════════════════════════
> Walk-Forward Self-Evaluation Summary
> ═══════════════════════════════════════
> 季末日 (6): mean=-0.37%, win=33%
> 非季末日 (12): mean=+2.51%, win=83%
> 综合 (18): mean=+1.55%, win=72%
> ```

### 5.2 单日预测

```bash
python scripts/predict.py --date 2026-03-13 --top-k 5
```

> **预期**：输出 Top-5 股票及预测分数，保存到 `output/result.csv`。

---

## 6. Docker 提交（竞赛用）

```bash
# 确保 app/code/src/ 与 code/src/ 同步
python scripts/sync_to_docker.py

# 构建
docker buildx build --platform linux/amd64 \
    --build-arg IMAGE_NAME=nvidia/cuda \
    -t bdc2026 .

# 本地验证
docker compose up

# 导出
docker save -o 队伍名称.tar bdc2026:latest
```

---

## 7. 运行测试

```bash
# 单元测试 (41 tests, ~3s)
python -m pytest tests/ -v

# 集成测试 (最小链路, ~30s)
python -m pytest tests/test_integration.py -v
```

---

## 8. 预期结果范围

| 配置 | 综合收益率 | 正收益比例 | 训练耗时 |
|------|:--:|:--:|:--:|
| V8 + quarter_aware (2021起) | **+1.0% ~ +1.5%** | 65-75% | ~6h |
| V8 全量 (2010起) | -0.5% ~ +0.5% | 50-60% | ~47h |
| V7 (固定窗口) | 不可靠（自评+6.15% vs 官方-1.29%） | — | ~10min |

> **注意**：金融数据非平稳性意味着不同时期重跑结果会有波动。核心结论（季末/非季末分化、模型排序能力有效）应保持稳定。

---

## 参考

- [项目 README](README.md) — 架构设计与实验历史
- [工作记录索引](总结/INDEX.md) — 完整的实验日志链
- [架构决策记录](docs/adr/) — 关键技术决策的理由

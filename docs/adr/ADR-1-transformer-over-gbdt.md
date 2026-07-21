# ADR-1: Transformer 而非纯 GBDT 作为主排序模型

**日期**: 2026-07-20  
**状态**: ✅ 已采纳  
**决策者**: 项目组（基于 07-07 GBDT Stacking 实验）

---

## 背景

在实验报告中讨论了替代排序算法后，需要决定主模型采用 Transformer 还是 GBDT（LightGBM/XGBoost LambdaRank）。

## 选项

### 选项 A: 纯 GBDT + 时序汇总（策略 B）
对 60 天序列做统计汇总（均值/标准差/趋势/分位数等），送入 LightGBM LambdaRank。

### 选项 B: Transformer（StockTransformer）
端到端的 Cross-Stock Attention + Temporal Encoder，直接处理原始时序。

### 选项 C: Transformer 表征 + GBDT Stacking（策略 D）
加载预训练 Transformer，提取中间表征，送入 LightGBM LambdaRank。

## 实验结果

| 方法 | 5 日收益率 | 结论 |
|------|:--:|------|
| 官方基线 | +2.52% | 简单策略 |
| V7 Transformer (收益门控) | **+6.15%** | 当前最优 |
| GBDT Stacking | **-5.32%** | 显著劣于纯 Transformer |

GBDT Stacking 的 NDCG@5 = 0.348，仅略好于随机排序（0.25~0.30）。

## 决策

**选择选项 B（纯 Transformer）。**

## 理由

1. **特征提取断点不当**：从 `ranking_layers` 输出提取的 128 维表征高度依赖后续 `score_head` 才能转化为有效排序，GBDT 无法利用这种"协作训练"的依赖关系。

2. **LambdaRank 离散化损失信息**：将连续超额收益转为 0~9 等级丢失了收益大小的细粒度信息，而这些信息正是 Transformer 排序损失利用的核心。

3. **忽视了时序交互**：GBDT 每个样本独立，无法利用 Cross-Stock Attention 学到的股票间相对关系。

4. **特征重要性高度集中**：Top-3 特征贡献了 50%+ 的 gain，大量特征未被有效利用。

## 后果

- Transformer 成为唯一主模型路线，后续所有优化（V7/V8/V9）均基于此架构
- GBDT 仅保留为对比基线，不再作为主方向迭代
- 纯策略 B（时序汇总 + 无 Transformer 特征）未实施，概率较低但存在被遗漏的风险

## 参考

- [2026-07-07 工作记录](../总结/2026-07-07-工作记录.md) — GBDT Stacking 实验详情
- `gbdt_stacking.py` — 实验脚本

#!/usr/bin/env python3
"""V5 训练结果全面分析"""

import pandas as pd
import numpy as np
import json
import re
import math
from scipy import stats
from collections import Counter

# ========== 0. 检查 reward 是否"坏掉" ==========
print("=" * 60)
print("0. Reward 健康检查")
print("=" * 60)

# 读取 debug log
debug_log_path = "/workspace/reward_debug.log"
scores = []
buffer_ratios = []
works = []
consumptions = []
work_targets = []
barrier_thresholds = []
overwork_pens = []
layer2_pens = []
macro_rewards = []
work_contribs = []

try:
    with open(debug_log_path, 'r') as f:
        content = f.read()
    
    # 解析 debug log
    pattern = r'\[(\d+)\].*?buf=([0-9.]+).*?work=([0-9.]+).*?tgt=([0-9.]+).*?bar=([0-9.]+).*?wr=([0-9.-]+).*?owp=([0-9.-]+).*?l2=([0-9.-]+).*?macro=([0-9.-]+).*?total=([0-9.-]+)'
    matches = re.findall(pattern, content)
    
    for m in matches:
        idx, buf, work, tgt, bar, wr, owp, l2, macro, total = m
        buffer_ratios.append(float(buf))
        works.append(float(work))
        work_targets.append(float(tgt))
        barrier_thresholds.append(float(bar))
        overwork_pens.append(float(owp))
        layer2_pens.append(float(l2))
        macro_rewards.append(float(macro))
        scores.append(float(total))
        work_contribs.append(float(wr))
    
    print(f"从 debug log 解析了 {len(scores)} 条记录")
except Exception as e:
    print(f"读取 debug log 失败: {e}")

# 读取验证集
val_path = "/workspace/QWEN2.5_42_GRPO_700step-/QWEN2.5_42_GRPO_1/data/verl_dataset_small/val.parquet"
df = pd.read_parquet(val_path)

def parse_extra_info(e):
    if isinstance(e, str):
        try:
            return json.loads(e)
        except:
            return {}
    return e if isinstance(e, dict) else {}

df['extra_parsed'] = df['extra_info'].apply(parse_extra_info)
df['buffer_ratio'] = df['extra_parsed'].apply(lambda x: x.get('buffer_ratio', None))
df['regime'] = df['extra_parsed'].apply(lambda x: x.get('regime', 'normal'))
df['regime_strength'] = df['extra_parsed'].apply(lambda x: x.get('regime_strength', None))
df['dpi'] = df['extra_parsed'].apply(lambda x: x.get('dpi', None))

# 从训练日志获取验证分数
train_log_path = "/workspace/train_v5_reward.log"
val_scores = []
val_works = []
val_cons = []

try:
    with open(train_log_path, 'r') as f:
        log_content = f.read()
    
    # 提取验证时的输出
    # 寻找 JSON 输出
    json_pattern = r'\{"work":\s*([0-9.]+),\s*"consumption":\s*([0-9.]+)\}'
    json_matches = re.findall(json_pattern, log_content)
    
    for w, c in json_matches[-200:]:  # 取最后 200 个（验证集大小）
        val_works.append(float(w))
        val_cons.append(float(c))
    
    # 提取 reward 均值
    reward_pattern = r"val-core/econ_agent/reward/mean@1[':]\s*np\.float64\(([0-9.-]+)\)"
    reward_matches = re.findall(reward_pattern, log_content)
    if reward_matches:
        print(f"验证 reward 均值: {reward_matches[-1]}")
    
    # 提取分数分布
    score_pattern = r"critic/score/(mean|max|min):([0-9.-]+)"
    score_matches = re.findall(score_pattern, log_content)
    if score_matches:
        score_dict = {k: float(v) for k, v in score_matches[-3:]}
        print(f"Score 分布: mean={score_dict.get('mean', 'N/A'):.4f}, "
              f"max={score_dict.get('max', 'N/A'):.4f}, "
              f"min={score_dict.get('min', 'N/A'):.4f}")

except Exception as e:
    print(f"读取训练日志失败: {e}")

print(f"\n从验证输出解析了 {len(val_works)} 个 work 值, {len(val_cons)} 个 consumption 值")

# 检查 NaN/Inf
print("\n--- NaN/Inf 检查 ---")
br_valid = df['buffer_ratio'].dropna()
print(f"buffer_ratio: 有效 {len(br_valid)}/{len(df)}, NaN/Inf: {len(df) - len(br_valid)}")

rs_valid = df['regime_strength'].dropna()
print(f"regime_strength: 有效 {len(rs_valid)}/{len(df)}, NaN/Inf: {len(df) - len(rs_valid)}")

dpi_valid = df['dpi'].dropna()
print(f"dpi: 有效 {len(dpi_valid)}/{len(df)}, NaN/Inf: {len(df) - len(dpi_valid)}")

# 负分占比（从 debug log）
if scores:
    neg_ratio = sum(1 for s in scores if s < 0) / len(scores)
    print(f"\n--- 负分占比 ---")
    print(f"负分 (<0): {neg_ratio*100:.1f}%")
    print(f"接近零 (-0.1~0.1): {sum(1 for s in scores if -0.1 <= s <= 0.1)/len(scores)*100:.1f}%")
    print(f"高分 (>0.3): {sum(1 for s in scores if s > 0.3)/len(scores)*100:.1f}%")

# Clip 命中率
if scores:
    clip_low = sum(1 for s in scores if s <= -0.99) / len(scores)
    clip_high = sum(1 for s in scores if s >= 0.99) / len(scores)
    print(f"\n--- Clip 命中率 ---")
    print(f"被 clip 到 -1: {clip_low*100:.2f}%")
    print(f"被 clip 到 +1: {clip_high*100:.2f}%")

# ========== 1. 动作锁死检查 ==========
print("\n" + "=" * 60)
print("1. 动作分布分析")
print("=" * 60)

if val_works:
    works_arr = np.array(val_works)
    cons_arr = np.array(val_cons)
    
    # 唯一值数量
    work_unique = len(set(val_works))
    cons_unique = len(set(val_cons))
    pairs = list(zip(val_works, val_cons))
    pair_unique = len(set(pairs))
    
    print(f"Work 唯一值数量: {work_unique}")
    print(f"Consumption 唯一值数量: {cons_unique}")
    print(f"动作组合数: {pair_unique}")
    
    # Top-K 分析
    print("\n--- Work Top-5 分布 ---")
    work_counter = Counter(val_works)
    for val, cnt in work_counter.most_common(5):
        print(f"  work={val:.2f}: {cnt} 次 ({cnt/len(val_works)*100:.1f}%)")
    
    print("\n--- Consumption Top-5 分布 ---")
    cons_counter = Counter(val_cons)
    for val, cnt in cons_counter.most_common(5):
        print(f"  cons={val:.2f}: {cnt} 次 ({cnt/len(val_cons)*100:.1f}%)")
    
    print("\n--- 动作组合 Top-5 ---")
    pair_counter = Counter(pairs)
    for (w, c), cnt in pair_counter.most_common(5):
        print(f"  (work={w:.2f}, cons={c:.2f}): {cnt} 次 ({cnt/len(pairs)*100:.1f}%)")
    
    # 基本统计
    print(f"\n--- Work 统计 ---")
    print(f"  mean={works_arr.mean():.3f}, std={works_arr.std():.3f}")
    print(f"  min={works_arr.min():.2f}, max={works_arr.max():.2f}")
    
    print(f"\n--- Consumption 统计 ---")
    print(f"  mean={cons_arr.mean():.3f}, std={cons_arr.std():.3f}")
    print(f"  min={cons_arr.min():.2f}, max={cons_arr.max():.2f}")

# ========== 2. Work vs Buffer Ratio ==========
print("\n" + "=" * 60)
print("2. 核心微观逻辑: Work vs Buffer Ratio")
print("=" * 60)

# 使用验证集的 buffer_ratio
br_values = df['buffer_ratio'].values
if len(val_works) == len(br_values):
    # 相关性分析
    valid_mask = ~np.isnan(br_values)
    br_clean = br_values[valid_mask]
    work_clean = np.array(val_works)[valid_mask]
    
    if len(br_clean) > 10:
        pearson_r, pearson_p = stats.pearsonr(br_clean, work_clean)
        spearman_r, spearman_p = stats.spearmanr(br_clean, work_clean)
        
        print(f"Pearson r = {pearson_r:.4f} (p={pearson_p:.4f})")
        print(f"Spearman ρ = {spearman_r:.4f} (p={spearman_p:.4f})")
        
        if pearson_r < -0.1:
            print("✅ 负相关方向正确")
        elif pearson_r > 0.1:
            print("⚠️ 正相关，与预期相反")
        else:
            print("⚠️ 相关性接近 0，微观逻辑未体现")
    
    # 分组均值
    print("\n--- 分组均值 (BR 三段) ---")
    low_mask = br_values < 2
    mid_mask = (br_values >= 2) & (br_values <= 3.5)
    high_mask = br_values > 3.5
    
    work_arr = np.array(val_works)
    
    low_work = work_arr[low_mask].mean() if low_mask.sum() > 0 else np.nan
    mid_work = work_arr[mid_mask].mean() if mid_mask.sum() > 0 else np.nan
    high_work = work_arr[high_mask].mean() if high_mask.sum() > 0 else np.nan
    
    print(f"  BR < 2:     work 均值 = {low_work:.3f} (n={low_mask.sum()})")
    print(f"  BR 2~3.5:   work 均值 = {mid_work:.3f} (n={mid_mask.sum()})")
    print(f"  BR > 3.5:   work 均值 = {high_work:.3f} (n={high_mask.sum()})")
    
    if low_work > mid_work > high_work:
        print("✅ Work 随 BR 递减，符合预期")
    else:
        print("⚠️ Work 未随 BR 递减")
    
    # 高 BR 错例
    very_high_br = br_values > 4
    high_work_mask = work_arr >= 0.8
    error_mask = very_high_br & high_work_mask
    error_rate = error_mask.sum() / very_high_br.sum() if very_high_br.sum() > 0 else 0
    print(f"\n--- 高 BR 错例 ---")
    print(f"  BR > 4 且 work >= 0.8: {error_mask.sum()} / {very_high_br.sum()} ({error_rate*100:.1f}%)")

# ========== 3. Consumption vs Regime ==========
print("\n" + "=" * 60)
print("3. 核心宏观逻辑: Consumption vs Regime")
print("=" * 60)

regime_values = df['regime'].values
if len(val_cons) == len(regime_values):
    cons_arr = np.array(val_cons)
    
    print("\n--- 各 Regime 消费统计 ---")
    regimes = ['recession', 'normal', 'boom']
    regime_stats = {}
    
    for r in regimes:
        mask = regime_values == r
        if mask.sum() > 0:
            cons_r = cons_arr[mask]
            regime_stats[r] = {
                'mean': cons_r.mean(),
                'std': cons_r.std(),
                'n': mask.sum()
            }
            print(f"  {r:10s}: mean={cons_r.mean():.3f}, std={cons_r.std():.3f}, n={mask.sum()}")
    
    # Boom vs Recession 差异
    if 'boom' in regime_stats and 'recession' in regime_stats:
        diff = regime_stats['boom']['mean'] - regime_stats['recession']['mean']
        print(f"\n  Boom - Recession 差异: {diff:+.4f}")
        
        if diff > 0.03:
            print("✅ Boom 消费更高，符合预期")
        elif diff > 0:
            print("⚠️ 方向正确但差异较小 (<0.03)")
        else:
            print("❌ Recession 消费更高，与预期相反")
    
    # 样本不均衡检查
    print("\n--- Regime 样本分布 ---")
    for r in regimes:
        cnt = (regime_values == r).sum()
        print(f"  {r}: {cnt} ({cnt/len(regime_values)*100:.1f}%)")

# ========== 4. 低分样本 Top-K 诊断 ==========
print("\n" + "=" * 60)
print("4. 低分样本 Top-10 诊断")
print("=" * 60)

if scores and len(val_works) >= 10:
    # 合并数据
    n = min(len(scores), len(val_works), len(br_values))
    
    data = []
    for i in range(n):
        data.append({
            'idx': i,
            'score': scores[i] if i < len(scores) else np.nan,
            'work': val_works[i] if i < len(val_works) else np.nan,
            'cons': val_cons[i] if i < len(val_cons) else np.nan,
            'br': br_values[i] if i < len(br_values) else np.nan,
            'regime': regime_values[i] if i < len(regime_values) else 'unknown',
            'work_target': work_targets[i] if i < len(work_targets) else np.nan,
            'barrier': barrier_thresholds[i] if i < len(barrier_thresholds) else np.nan,
            'overwork_pen': overwork_pens[i] if i < len(overwork_pens) else np.nan,
            'layer2_pen': layer2_pens[i] if i < len(layer2_pens) else np.nan,
            'macro_reward': macro_rewards[i] if i < len(macro_rewards) else np.nan,
        })
    
    # 按分数排序，取最低 10 个
    data_sorted = sorted(data, key=lambda x: x['score'] if not np.isnan(x['score']) else 999)
    
    print("最低分 10 个样本：")
    print("-" * 80)
    for i, d in enumerate(data_sorted[:10]):
        print(f"[{i+1}] Score={d['score']:.3f} | BR={d['br']:.2f} | Regime={d['regime']}")
        print(f"    Work={d['work']:.2f} (tgt={d['work_target']:.2f}, bar={d['barrier']:.2f})")
        print(f"    Cons={d['cons']:.2f}")
        print(f"    Penalties: overwork={d['overwork_pen']:.3f}, l2={d['layer2_pen']:.3f}, macro={d['macro_reward']:.3f}")
        print()
    
    # 统计低分样本的特征
    low_score_data = data_sorted[:20]
    low_regimes = [d['regime'] for d in low_score_data]
    low_brs = [d['br'] for d in low_score_data if not np.isnan(d['br'])]
    low_works = [d['work'] for d in low_score_data if not np.isnan(d['work'])]
    
    print("\n--- 低分样本特征统计 ---")
    print(f"Regime 分布: {Counter(low_regimes)}")
    print(f"BR 均值: {np.mean(low_brs):.2f}")
    print(f"Work 均值: {np.mean(low_works):.2f}")

# ========== 5. 约束触发率 ==========
print("\n" + "=" * 60)
print("5. 约束触发率")
print("=" * 60)

if val_works and val_cons:
    work_arr = np.array(val_works)
    cons_arr = np.array(val_cons)
    
    # Work/Cons 越界
    work_out = ((work_arr < 0) | (work_arr > 1)).sum()
    cons_out = ((cons_arr < 0) | (cons_arr > 1)).sum()
    print(f"Work 越界 (<0 或 >1): {work_out} ({work_out/len(work_arr)*100:.2f}%)")
    print(f"Cons 越界 (<0 或 >1): {cons_out} ({cons_out/len(cons_arr)*100:.2f}%)")
    
    # Extreme penalty
    extreme_work = ((work_arr < 0.05) | (work_arr > 0.95)).sum()
    extreme_cons = ((cons_arr < 0.05) | (cons_arr > 0.95)).sum()
    print(f"\nExtreme Work (<0.05 或 >0.95): {extreme_work} ({extreme_work/len(work_arr)*100:.1f}%)")
    print(f"Extreme Cons (<0.05 或 >0.95): {extreme_cons} ({extreme_cons/len(cons_arr)*100:.1f}%)")
    
    # Overconsume
    overconsume = (cons_arr > 0.90).sum()
    print(f"\nOverconsume (cons > 0.90): {overconsume} ({overconsume/len(cons_arr)*100:.1f}%)")
    
    # Overwork Layer2
    overwork_l2 = (work_arr > 0.86).sum()
    print(f"Overwork Layer2 (work > 0.86): {overwork_l2} ({overwork_l2/len(work_arr)*100:.1f}%)")

# ========== 6. 选择建议 ==========
print("\n" + "=" * 60)
print("6. Checkpoint 选择建议")
print("=" * 60)

print("\n评估维度：")
print("-" * 40)

# 1. Work-BR 相关性
if 'pearson_r' in dir():
    if pearson_r < -0.1:
        print("✅ Work-BR 负相关: 是")
    else:
        print("❌ Work-BR 负相关: 否")

# 2. Regime-Consumption 方向
if 'diff' in dir():
    if diff > 0:
        print("✅ Boom > Recession 消费: 是")
    else:
        print("❌ Boom > Recession 消费: 否")

# 3. 动作锁死
if val_works:
    top1_ratio = work_counter.most_common(1)[0][1] / len(val_works)
    if top1_ratio < 0.5:
        print(f"✅ 动作未锁死: Top-1 占比 {top1_ratio*100:.1f}%")
    else:
        print(f"❌ 动作锁死: Top-1 占比 {top1_ratio*100:.1f}%")

# 4. 平均分
if scores:
    avg_score = np.mean(scores)
    print(f"📊 平均分: {avg_score:.3f}")

print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)


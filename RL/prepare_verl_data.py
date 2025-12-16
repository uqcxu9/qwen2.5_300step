# prepare_verl_data.py
import pickle as pkl
import pandas as pd
import numpy as np
import os
import sys
import json

# 添加 ai_economist 模块路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 经济决策 agent 的系统提示
SYSTEM_PROMPT = """You are an economic decision-making agent. Based on your current financial situation, you need to make two decisions:
1. work: A value between 0 and 1 representing your labor supply (0 = no work, 1 = full-time work)
2. consumption: A value between 0 and 1 representing the proportion of your disposable income to consume

You MUST respond with a valid JSON object in this exact format:
{"work": <float 0-1>, "consumption": <float 0-1>}

Do not include any other text or explanation, only output the JSON."""


def compute_macro_indicators(dense_log, prices):
    """
    预计算所有月份的宏观指标
    
    参数:
        dense_log: 模拟日志
        prices: 价格列表（从 prepare_verl_dataset 传入）
    
    返回:
        macro_data: dict, 包含每个月的失业率、GDP增长率、价格通胀率
    """
    states = dense_log['states']
    actions = dense_log['actions']
    
    num_months = len(states)
    num_agents = 100
    
    # ========== 1. 月度失业率 ==========
    # 与 unemployment_monthly.py 一致
    monthly_unemployment = []
    for m in range(num_months):
        unemployed = 0
        employed = 0
        
        for agent_id_str, agent_state in states[m].items():
            if (not agent_id_str.isdigit()) or (not isinstance(agent_state, dict)):
                continue
            
            job = agent_state.get("endogenous", {}).get("job")
            if job == "Unemployment":
                unemployed += 1
            else:
                employed += 1
        
        labor_force = employed + unemployed
        rate = unemployed / labor_force if labor_force > 0 else 0
        monthly_unemployment.append(rate)
    
    # ========== 2. 月度供给（用于GDP计算）==========
    # 与 gdp_manual_yearly.py 一致: S = Σ(l_j × 168 × A)
    monthly_supply = []
    A = 1  # 生产率
    num_labor_hours = 168
    
    for m in range(len(actions)):
        month_actions = actions[m] or {}
        total_supply = 0
        
        for agent_id, action in month_actions.items():
            if agent_id == 'p':
                continue
            
            if isinstance(action, (list, tuple)) and len(action) >= 1:
                labor = int(action[0])
            elif isinstance(action, dict):
                labor = int(action.get('SimpleLabor', 0))
            else:
                labor = 0
            
            total_supply += labor * num_labor_hours * A
        
        monthly_supply.append(total_supply)
    
    # ========== 3. 价格序列（直接使用传入的 prices）==========
    # 确保长度匹配
    prices = list(prices)  # 创建副本，避免修改原始列表
    while len(prices) < num_months:
        prices.append(prices[-1] if prices else 1.0)
    
    # ========== 4. 计算月度GDP = S × P ==========
    monthly_gdp = []
    for m in range(min(len(monthly_supply), len(prices))):
        gdp = monthly_supply[m] * prices[m]
        monthly_gdp.append(gdp)
    
    # ========== 5. 计算年同比指标 ==========
    macro_data = {
        'unemployment_rate': [],  # 月度失业率（小数）
        'gdp_growth': [],         # 年同比GDP增长率（百分比）
        'price_inflation': [],    # 年同比价格通胀率（百分比）
    }
    
    for m in range(num_months):
        # 5.1 失业率：当月数据
        unemp = monthly_unemployment[m] if m < len(monthly_unemployment) else 0.0
        macro_data['unemployment_rate'].append(unemp)
        
        # 5.2 价格通胀：年同比
        if m >= 12 and prices[m-12] > 0:
            price_inflation = (prices[m] - prices[m-12]) / prices[m-12] * 100
        else:
            # 前12个月用月环比（或设为0）
            if m > 0 and prices[m-1] > 0:
                price_inflation = (prices[m] - prices[m-1]) / prices[m-1] * 100 * 12  # 年化
            else:
                price_inflation = 0.0
        macro_data['price_inflation'].append(price_inflation)
        
        # 5.3 GDP增长：年同比（需要24个月数据才能正确计算）
        if m >= 24:
            gdp_recent_12m = sum(monthly_gdp[m-11:m+1])   # 第 m-11 到 m 月
            gdp_prev_12m = sum(monthly_gdp[m-23:m-11])    # 第 m-23 到 m-12 月
            
            if gdp_prev_12m > 0:
                gdp_growth = (gdp_recent_12m - gdp_prev_12m) / gdp_prev_12m * 100
            else:
                gdp_growth = 0.0
        else:
            gdp_growth = 0.0  # 前24个月数据不足，设为0
        
        macro_data['gdp_growth'].append(gdp_growth)
    
    print(f"✓ 计算宏观指标完成:")
    print(f"  - 失业率范围: [{min(macro_data['unemployment_rate'])*100:.2f}%, {max(macro_data['unemployment_rate'])*100:.2f}%]")
    print(f"  - GDP增长范围: [{min(macro_data['gdp_growth']):.2f}%, {max(macro_data['gdp_growth']):.2f}%]")
    print(f"  - 通胀范围: [{min(macro_data['price_inflation']):.2f}%, {max(macro_data['price_inflation']):.2f}%]")
    
    return macro_data, prices


def prepare_verl_dataset(data_dir, output_dir, num_agents=100):
    """
    从dense_log提取verl训练数据
    包含微观变量 + 宏观变量 + MPC所需的前一步数据
    """
    
    dense_log_path = f"{data_dir}/dense_log.pkl"
    with open(dense_log_path, 'rb') as f:
        dense_log = pkl.load(f)
    
    # ========== 加载价格数据 ==========
    env_path = f"{data_dir}/env_240.pkl"
    prices = None
    
    if os.path.exists(env_path):
        try:
            # 直接加载（环境有 ai_economist 包）
            with open(env_path, "rb") as f:
                env_data = pkl.load(f)
            if hasattr(env_data, 'world') and hasattr(env_data.world, 'price'):
                prices = [float(p) for p in env_data.world.price]
                print(f"✓ 从 env_240.pkl 获取价格成功，共 {len(prices)} 个月")
                print(f"  价格范围: [{min(prices):.2f}, {max(prices):.2f}]")
        except Exception as e:
            print(f"⚠ 加载 env_240.pkl 失败: {e}")
    
    if prices is None:
        print("⚠ 无法获取价格，使用常数价格 1.0（通胀将为 0%）")
        prices = [1.0] * len(dense_log['states'])
    
    states = dense_log['states']
    actions = dense_log['actions']
    periodic_tax = dense_log['PeriodicTax']
    prompts = dense_log.get('prompts', [])
    
    # 预计算宏观指标（传入价格列表）
    macro_data, prices = compute_macro_indicators(dense_log, prices)
    
    # ====== regime thresholds (rolling quantiles, adaptive) ======
    U = np.array(macro_data["unemployment_rate"], dtype=float)   # 0~1
    G = np.array(macro_data["gdp_growth"], dtype=float)          # %
    PI = np.array(macro_data["price_inflation"], dtype=float)    # %

    WINDOW = 24  # 2 years
    q_low, q_high = 0.20, 0.80  # 20/80 分位做"极端"判定

    def rolling_q(arr, t, q):
        s = max(0, t - WINDOW + 1)
        return float(np.quantile(arr[s:t+1], q))

    def clip01(x):
        return float(max(0.0, min(1.0, x)))
    
    # 预计算每个agent每个月的DPI和消费金额（用于MPC）
    agent_history = {}  # agent_history[agent_id][t] = {'dpi': ..., 'consumption_amount': ...}
    
    for t in range(len(states)):
        for agent_id in range(num_agents):
            agent_id_str = str(agent_id)
            
            if agent_id_str not in states[t]:
                continue
            
            state = states[t][agent_id_str]
            
            try:
                income = state['income']['Coin']
                wealth = state['inventory']['Coin']
                consumption_amount = state['consumption']['Coin']
            except (KeyError, TypeError):
                continue
            
            if t < len(periodic_tax):
                tax_info = periodic_tax[t].get(agent_id_str, {})
            else:
                tax_info = {}
            tax_paid = tax_info.get('tax_paid', 0)
            lump_sum = tax_info.get('lump_sum', 0)
            
            dpi = income + lump_sum - tax_paid
            
            if agent_id not in agent_history:
                agent_history[agent_id] = {}
            
            agent_history[agent_id][t] = {
                'dpi': dpi,
                'consumption_amount': consumption_amount,
            }
    
    samples = []
    
    # 错误统计
    errors = {
        'missing_state': 0,
        'missing_prompt': 0,
        'missing_action': 0,
        'invalid_action_format': 0,
    }
    
    for t in range(1, len(actions)):
        for agent_id in range(num_agents):
            agent_id_str = str(agent_id)
            
            # 检查 state 是否存在
            if agent_id_str not in states[t]:
                errors['missing_state'] += 1
                continue
            
            # 检查 prompt 是否存在
            if t >= len(prompts) or agent_id_str not in prompts[t]:
                errors['missing_prompt'] += 1
                continue
            
            raw_prompt = prompts[t][agent_id_str]
            if not raw_prompt or not isinstance(raw_prompt, str) or len(raw_prompt) < 100:
                errors['missing_prompt'] += 1
                continue
            
            state = states[t][agent_id_str]
            
            # 检查必要的 state 字段
            try:
                income = state['income']['Coin']
                wealth = state['inventory']['Coin']
                skill = state['skill']
                consumption_amount = state['consumption']['Coin']
            except (KeyError, TypeError):
                errors['missing_state'] += 1
                continue
            
            if t < len(periodic_tax):
                tax_info = periodic_tax[t].get(agent_id_str, {})
            else:
                tax_info = {}
            tax_paid = tax_info.get('tax_paid', 0)
            lump_sum = tax_info.get('lump_sum', 0)
            
            # 微观变量计算（与分析脚本一致）
            dpi = income + lump_sum - tax_paid
            cash_on_hand = wealth + dpi
            buffer_ratio = cash_on_hand / (dpi + 1e-8) if dpi > 1e-6 else 1.0
            
            # 获取前一步数据（用于MPC）
            prev_dpi = -1.0
            prev_consumption_amount = -1.0
            
            if agent_id in agent_history and (t-1) in agent_history[agent_id]:
                prev_data = agent_history[agent_id][t-1]
                prev_dpi = prev_data['dpi']
                prev_consumption_amount = prev_data['consumption_amount']
            
            # 获取宏观变量
            unemployment_rate = macro_data['unemployment_rate'][t] if t < len(macro_data['unemployment_rate']) else 0.0
            gdp_growth = macro_data['gdp_growth'][t] if t < len(macro_data['gdp_growth']) else 0.0
            price_inflation = macro_data['price_inflation'][t] if t < len(macro_data['price_inflation']) else 0.0
            
            # ====== compute regime + strength at month t (adaptive) ======
            u = float(unemployment_rate)
            g = float(gdp_growth)
            pi = float(price_inflation)

            # 前24个月 gdp_growth 全是0，会污染判定，强制 normal
            if t < 24:
                regime = "normal"
                regime_strength = 0.15
            else:
                # rolling thresholds
                u_lo = rolling_q(U, t, q_low)
                u_hi = rolling_q(U, t, q_high)
                g_lo = rolling_q(G, t, q_low)
                g_hi = rolling_q(G, t, q_high)
                
                # 用分位区间宽度做归一化（更稳定）
                g_span  = max(rolling_q(G, t, 0.8) - rolling_q(G, t, 0.2), 1e-6)
                u_span  = max(rolling_q(U, t, 0.8) - rolling_q(U, t, 0.2), 1e-6)

                # regime classification（暂不做 stagflation，先跑通三类）
                if (g <= g_lo) and (u >= u_hi):
                    regime = "recession"
                elif (g >= g_hi) and (u <= u_lo):
                    regime = "boom"
                else:
                    regime = "normal"

                # strength: 用分位区间宽度归一化，recession/boom 给底噪 0.2
                if regime == "recession":
                    s_g = (g_lo - g) / g_span if g < g_lo else 0.0
                    s_u = (u - u_hi) / u_span if u > u_hi else 0.0
                    regime_strength = clip01(0.2 + 0.8 * (0.5 * s_g + 0.5 * s_u))

                elif regime == "boom":
                    s_g = (g - g_hi) / g_span if g > g_hi else 0.0
                    s_u = (u_lo - u) / u_span if u < u_lo else 0.0
                    regime_strength = clip01(0.2 + 0.8 * (0.5 * s_g + 0.5 * s_u))

                else:
                    regime_strength = 0.15  # normal 给很小权重
            
            # 检查 action
            action_data = actions[t].get(agent_id_str)
            if action_data is None:
                errors['missing_action'] += 1
                continue

            if isinstance(action_data, dict):
                work = action_data.get('SimpleLabor')
                cons_idx = action_data.get('SimpleConsumption')
                if work is None or cons_idx is None:
                    errors['invalid_action_format'] += 1
                    continue
            elif isinstance(action_data, (list, tuple)) and len(action_data) >= 2:
                work = action_data[0]
                cons_idx = action_data[1]
            else:
                errors['invalid_action_format'] += 1
                continue

            consumption_prop = cons_idx * 0.02
            
            # 构建 verl 期望的 chat 格式 prompt
            chat_prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_prompt}
            ]
            
            extra_info_dict = {
                # 基本信息
                "timestep": t,
                "agent_id": agent_id,
                
                # 微观变量（与分析脚本一致）
                "income": float(income),
                "wealth": float(wealth),
                "tax_paid": float(tax_paid),
                "lump_sum": float(lump_sum),
                "dpi": float(dpi),
                "buffer_ratio": float(buffer_ratio),
                "skill": float(skill),
                
                # MPC所需的前一步数据
                "prev_dpi": float(prev_dpi),
                "prev_consumption_amount": float(prev_consumption_amount),
                
                # 宏观变量
                "unemployment_rate": float(unemployment_rate),      # 小数，如0.08
                "gdp_growth": float(gdp_growth),                    # 百分比，如1.5
                "price_inflation": float(price_inflation),          # 百分比，如2.0
                
                # Regime（宏观周期）
                "regime": regime,
                "regime_strength": float(regime_strength),
                
                # Ground truth（用于分析）
                "gt_work": int(work),
                "gt_consumption": float(consumption_prop),
            }
            
            samples.append({
                "prompt": chat_prompt,
                "data_source": "econ_agent",
                "extra_info": extra_info_dict,
                "reward_model": {"ground_truth": ""},
                "ability": "economic_decision",
            })
    
    # 打印统计
    print(f"\n=== 数据处理统计 ===")
    print(f"成功样本数: {len(samples)}")
    print(f"错误统计:")
    for err_type, count in errors.items():
        if count > 0:
            print(f"  - {err_type}: {count}")
    
    # Regime 分布统计（重要：确保不是 95% normal）
    from collections import Counter
    regime_cnt = Counter([s["extra_info"]["regime"] for s in samples])
    print(f"\n=== Regime 分布 ===")
    for r, c in sorted(regime_cnt.items()):
        pct = c / len(samples) * 100
        print(f"  - {r}: {c} ({pct:.1f}%)")
    
    if len(samples) == 0:
        raise ValueError("没有有效样本！请检查数据源。")
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(samples)
    
    # 按 agent_id 切分，保证 val 的 agent 不出现在 train
    VAL_AGENTS = 5  # 5个agent作为验证
    VAL_SAMPLE_SIZE = 200  # 验证集最终抽样到 200 条
    
    # 提取所有 agent_id（排序确保可复现）
    agent_ids = sorted(df["extra_info"].apply(lambda x: x["agent_id"]).unique())
    
    # 随机选择验证用的 agent
    rng = np.random.default_rng(42)
    val_agents = set(rng.choice(agent_ids, size=min(VAL_AGENTS, len(agent_ids)), replace=False))
    
    # 按 agent_id 划分
    val_mask = df["extra_info"].apply(lambda x: x["agent_id"] in val_agents)
    val_df_full = df[val_mask]
    train_df = df[~val_mask]
    
    # 分层抽样：每个 agent 抽固定数量，并尽量让 timestep 均匀覆盖
    if len(val_df_full) > VAL_SAMPLE_SIZE:
        samples_per_agent = VAL_SAMPLE_SIZE // VAL_AGENTS

        def uniform_time_sample(g, k, seed=42):
            # 取 timestep 列表并排序
            ts = g["extra_info"].apply(lambda x: x["timestep"]).to_numpy()
            order = np.argsort(ts)
            g_sorted = g.iloc[order].reset_index(drop=True)

            if len(g_sorted) <= k:
                return g_sorted

            # 等距取样索引（覆盖全时间段）
            idx = np.linspace(0, len(g_sorted) - 1, num=k)
            idx = np.round(idx).astype(int)
            idx = np.unique(idx)

            # 如果 unique 后不足 k，补随机
            if len(idx) < k:
                remain = np.setdiff1d(np.arange(len(g_sorted)), idx)
                rng = np.random.default_rng(seed)
                extra = rng.choice(remain, size=k - len(idx), replace=False)
                idx = np.concatenate([idx, extra])

            return g_sorted.iloc[idx]

        val_df = (val_df_full
                  .groupby(val_df_full["extra_info"].apply(lambda x: x["agent_id"]))
                  .apply(lambda g: uniform_time_sample(g, samples_per_agent, seed=42))
                  .reset_index(drop=True))
    else:
        val_df = val_df_full
    
    print(f"验证集 agent: {sorted(val_agents)}")
    print(f"验证集: {len(val_df_full)} 条 -> 分层抽样后 {len(val_df)} 条 (每agent {VAL_SAMPLE_SIZE // VAL_AGENTS} 条)")

    
    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/val.parquet", index=False)
    
    print(f"\n✅ Saved {len(train_df)} training, {len(val_df)} validation samples")
    print(f"   输出目录: {output_dir}")
    
    # 验证数据格式
    print(f"\n=== 数据格式验证 ===")
    sample = samples[100]  # 取一个非第一个的样本
    print(f"extra_info keys: {list(sample['extra_info'].keys())}")
    print(f"  unemployment_rate: {sample['extra_info']['unemployment_rate']:.4f}")
    print(f"  gdp_growth: {sample['extra_info']['gdp_growth']:.2f}%")
    print(f"  price_inflation: {sample['extra_info']['price_inflation']:.2f}%")
    print(f"  prev_dpi: {sample['extra_info']['prev_dpi']:.2f}")
    print(f"  prev_consumption_amount: {sample['extra_info']['prev_consumption_amount']:.2f}")
    
    return train_df, val_df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    data_dir = os.path.join(base_dir, "data/gpt-3-noperception-reflection-1-100agents-240months")
    output_dir = os.path.join(base_dir, "data/verl_dataset_small")
    
    prepare_verl_dataset(data_dir, output_dir)
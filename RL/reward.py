import json
import re
import math  


def range_reward(x: float, low: float, high: float) -> float:
    """
    改进版：区间内奖励更平缓，避免"随便落在区间里就行"
    - 在区间内：中间最好 (=1.0)，边界 (=0.5)
    - 超出区间：按距离扣分
    
    返回值范围：约 [-1, 1]
    """
    # 除零保护
    width = max(high - low, 1e-6)
    
    if x < low:
        return -min((low - x) / width, 1.0)
    elif x > high:
        return -min((x - high) / width, 1.0)
    else:
        mid = 0.5 * (low + high)
        half = max(0.5 * width, 1e-6)
        return 0.5 + 0.5 * (1.0 - abs(x - mid) / half)



def parse_action(response: str):
    """解析模型输出的 JSON"""
    try:
        text = response.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            return json.loads(json_match.group())
        return None
    except:
        return None


def _to_float_or_none(x):
    """将值转换为 float，如果是 None/NaN/Inf 则返回 None"""
    if x is None:
        return None
    try:
        x = float(x)
    except:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


_DEBUG_COUNT = 0
_DEFAULT_VALUE_COUNT = 0  # 追踪默认值触发次数


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float:
    global _DEBUG_COUNT, _DEFAULT_VALUE_COUNT
    _DEBUG_COUNT += 1

    # 调试日志（只记录前 5 个）
    if _DEBUG_COUNT <= 5:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(f"\n=== DEBUG {_DEBUG_COUNT} ===\n")
                f.write(f"solution_str: {repr(solution_str)[:500]}\n")
                f.write(f"extra_info: {repr(extra_info)[:300]}\n")
        except:
            pass

    if data_source != "econ_agent":
        return 0.0

    reward = 0.0

    # ========== 1. JSON 格式检查 ==========
    action = parse_action(solution_str)
    if action is None:
        return -1.0

    work = action.get("work")
    consumption = action.get("consumption")

    if work is None or consumption is None:
        return -0.8

    try:
        work = float(work)
        consumption = float(consumption)
    except:
        return -0.8

    # ========== 2. 范围检查（软约束）==========
    if not (0 <= work <= 1):
        reward -= 0.3
    if not (0 <= consumption <= 1):
        reward -= 0.3

    work = max(0.0, min(1.0, work))
    consumption = max(0.0, min(1.0, consumption))

    # ========== 3. 解析 extra_info ==========
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}
    elif extra_info is None:
        extra_info = {}

    # ------ 3.1 微观变量 ------
    income   = _to_float_or_none(extra_info.get('income', 0))   or 0.0
    lump_sum = _to_float_or_none(extra_info.get('lump_sum', 0)) or 0.0
    tax_paid = _to_float_or_none(extra_info.get('tax_paid', 0)) or 0.0
    wealth   = _to_float_or_none(extra_info.get('wealth', 0))   or 0.0

    dpi = _to_float_or_none(extra_info.get('dpi', None))
    if dpi is None:
        dpi = income + lump_sum - tax_paid
    dpi = max(dpi, 1.0)  # 避免除零

    buffer_ratio = _to_float_or_none(extra_info.get('buffer_ratio', None))
    if buffer_ratio is None:
        cash_on_hand = wealth + dpi
        try:
            buffer_ratio = cash_on_hand / dpi
        except Exception:
            buffer_ratio = 2.5
    buffer_ratio = max(0.0, min(10.0, float(buffer_ratio)))



    # ------ 3.2 宏观变量 ------
    unemp = _to_float_or_none(extra_info.get('unemployment_rate', None))
    gdp_g = _to_float_or_none(extra_info.get('gdp_growth', None))
    infl  = _to_float_or_none(extra_info.get('price_inflation', None))

    # ✅ 提前解析 regime + effective_strength，让 5.2 可按宏观强度调权/融合目标
    regime = extra_info.get("regime", None)
    if regime is None:
        regime = "normal"
        _DEFAULT_VALUE_COUNT += 1
        if _DEFAULT_VALUE_COUNT <= 10:
            try:
                with open('/workspace/reward_debug.log', 'a') as f:
                    f.write(f"[WARNING] Missing regime at sample {_DEBUG_COUNT}\n")
            except:
                pass

    regime_strength = _to_float_or_none(extra_info.get("regime_strength", None))
    if regime_strength is None:
        regime_strength = 0.15
        _DEFAULT_VALUE_COUNT += 1
        if _DEFAULT_VALUE_COUNT <= 10:
            try:
                with open('/workspace/reward_debug.log', 'a') as f:
                    f.write(f"[WARNING] Missing regime_strength at sample {_DEBUG_COUNT}\n")
            except:
                pass

    regime_strength = max(0.0, min(1.0, regime_strength))
    if regime == "normal":
        effective_strength = max(regime_strength, 0.30)
    else:
        effective_strength = max(regime_strength, 0.50)

    # ========== 4. Saving Rate ==========
    consumption_amount = consumption * dpi
    saving = dpi - consumption_amount
    saving_rate = saving / dpi  # 等于 1 - consumption

    # ========== 5. 评分计算 ==========
    # ------ 5.1 储蓄率约束（权重 0.10）------
    sr_reward = range_reward(saving_rate, 0.014, 0.60)
    reward += 0.07 * sr_reward

    # ------ 5.2 Work 行为偏好（权重 0.25）------
    # ✅ 常量定义
    WORK_MIN = 0.50

    # ✅ 连续目标：buffer_ratio 越大，work_target 越低（才能让相关性显著变负）
    br = buffer_ratio

    # piecewise linear target: br<=1.6 -> 0.90, br>=4.2 -> WORK_MIN，中间线性下降
    if br <= 1.6:
        work_target = 0.90
    elif br >= 4.2:
        work_target = WORK_MIN
    else:
        work_target = 0.90 + (WORK_MIN - 0.90) * (br - 1.6) / (4.2 - 1.6)

    # band: 给一个可容忍范围，避免学成单点；br 越极端，允许范围略窄
    work_band_br = 0.12 if 1.8 <= br <= 3.8 else 0.10

    # ✅ regime 侧 work target（用于和 buffer_ratio target 融合，避免 5.2/5.3 冲突）
    if regime == "recession":
        work_target_r = 0.85
    elif regime == "boom":
        work_target_r = 0.30
    else:
        work_target_r = 0.55

    # ✅ 融合系数：宏观越"强"（boom/recession 越明显），越听 regime
    # effective_strength≈0.30(normal)->alpha≈0.10；≈0.50(boom/recession)->alpha≈0.30
    alpha = max(0.0, min(0.6, (effective_strength - 0.20)))

    work_target_mix = (1.0 - alpha) * work_target + alpha * work_target_r

    work_reward = range_reward(
        work,
        max(0.0, work_target_mix - work_band_br),
        min(1.0, work_target_mix + work_band_br),
    )
    if work > work_target_mix:
        excess = (work - work_target_mix) / max(work_band_br, 1e-6)  # normalized
        work_reward -= 0.20 * min(excess ** 1.3, 3.0) 



    # ✅ 5.2 的权重按宏观强度"自动让路"，避免永远压过 regime
    w_work_micro = 0.25 * (1.0 - 0.6 * alpha)
    reward += w_work_micro * work_reward


    # ✅ 双层 Barrier：经济合理 + 破0.80盆地
    overwork_pen = 0.0

    BARRIER_OFFSET = 0.10
    BARRIER_K_MAIN = 0.12
   

    thr_follow = work_target_mix + BARRIER_OFFSET
    barrier_threshold = min(0.95, thr_follow)
    _dbg_barrier = barrier_threshold

    if work > barrier_threshold:
        overwork_pen -= BARRIER_K_MAIN * min((work - barrier_threshold) / 0.15, 1.0)

    # Layer 2: continuous overwork penalty (no special value like 0.80)
    GLOBAL_START = 0.86
    GLOBAL_SPAN = 0.14
    BARRIER_K_GLOBAL = 0.03

    layer2_pen = 0.0
    if work > GLOBAL_START:
        layer2_pen -= BARRIER_K_GLOBAL * min((work - GLOBAL_START) / GLOBAL_SPAN, 1.0)

    overwork_pen += layer2_pen

    reward += overwork_pen

    # ✅ 独立过度消费惩罚：防止 consumption 锁死到极端值
    overconsume_pen = 0.0
    if consumption > 0.90:
        overconsume_pen -= 0.20 * (consumption - 0.90) / 0.10  # 0.90->0, 1.00->-0.20
    reward += overconsume_pen

    # ===== 5.3(A) consumption target: micro + regime mix (avoid too hard) =====

    # 1) micro-side consumption target from buffer_ratio (br 越大，允许更高消费)
    if br <= 1.6:
        cons_target_br = 0.60   # 钱少 → 省着花
    elif br >= 4.2:
        cons_target_br = 0.78   # 钱多 → 可以多花
    else:
        cons_target_br = 0.60 + (0.78 - 0.60) * (br - 1.6) / (4.2 - 1.6)
     # 2) regime-side consumption target (仍然保留方向，但别太极端)
    if regime == "recession":
        cons_target_r = 0.62
    elif regime == "boom":
        cons_target_r = 0.80
    else:
        cons_target_r = 0.70

    # 3) mix: 宏观越强越听 regime，但上限不让它“压死”
    beta = max(0.0, min(0.6, (effective_strength - 0.20)))
    cons_target_mix = (1.0 - beta) * cons_target_br + beta * cons_target_r

    # 4) wider band: 给更多自由度，避免学成单点策略
    cons_band = 0.12 if regime == "normal" else 0.14

    cons_r = range_reward(
        consumption,
        max(0.0, cons_target_mix - cons_band),
        min(1.0, cons_target_mix + cons_band),
    )

    # action_struct 只用消费信号
    action_struct = cons_r


    # (B) anti-extreme brake（保留原逻辑；注意它会乘 regime_strength）
    extreme_pen = 0.0
    if work < 0.05 or work > 0.95:
        extreme_pen -= 0.10
    if consumption < 0.05 or consumption > 0.95:
        extreme_pen -= 0.10


    # (C) optional macro guardrail
    guard_parts = []
    guard_w = []

    if unemp is not None:
        guard_parts.append(range_reward(unemp, 0.02, 0.20))
        guard_w.append(0.40)
    if gdp_g is not None:
        guard_parts.append(range_reward(gdp_g, -5.0, 10.0))
        guard_w.append(0.35)
    if infl is not None:
        guard_parts.append(range_reward(infl, -2.0, 8.0))
        guard_w.append(0.25)

    if guard_w:
        wsum = sum(guard_w)
        guard = sum(p * w for p, w in zip(guard_parts, guard_w)) / wsum
    else:
        guard = 0.0

    guard = max(-1.0, min(1.0, guard))

    # (D) combine
    macro_reward = (0.9 * action_struct + 0.1 * guard + extreme_pen) * effective_strength


    reward += 0.65 * macro_reward

    # ========== 6. 调试日志 ==========
    if _DEBUG_COUNT <= 20:
        try:
            unemp_str = f"{unemp:.3f}" if unemp is not None else "None"
            gdp_str = f"{gdp_g:.2f}" if gdp_g is not None else "None"
            infl_str = f"{infl:.2f}" if infl is not None else "None"
            macro_used = ''.join([
                'pi' if infl is not None else '',
                'u'  if unemp is not None else '',
                'g'  if gdp_g is not None else ''
            ]) or 'none'
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(
                    f"[{_DEBUG_COUNT:03d}] "
                    f"sr={saving_rate:.3f} sr_r={sr_reward:.3f} sr_contrib={(0.07*sr_reward):.3f} | "
                    f"buf={buffer_ratio:.2f} work={work:.2f} "
                    f"tgt={work_target_mix:.2f} bar={_dbg_barrier:.2f} "
                    f"wr={work_reward:.2f} work_w={w_work_micro:.3f} work_contrib={(w_work_micro*work_reward):.3f} | "
                    f"owp={overwork_pen:.3f} l2={layer2_pen:.3f} owp_contrib={overwork_pen:.3f} | "
                    f"macro={macro_reward:.3f} macro_w={0.65:.3f} macro_contrib={(0.65*macro_reward):.3f} | "
                    f"total={reward:.3f}\n"

                )
        except:
            pass

    # 每 1000 个样本报告一次默认值触发情况
    if _DEBUG_COUNT % 1000 == 0 and _DEFAULT_VALUE_COUNT > 0:
        try:
            with open('/workspace/reward_debug.log', 'a') as f:
                f.write(
                    f"[STATS] Processed {_DEBUG_COUNT} samples, default value triggered {_DEFAULT_VALUE_COUNT} times "
                    f"({_DEFAULT_VALUE_COUNT/_DEBUG_COUNT*100:.2f}%)\n"
                )
        except:
            pass

    # ========== 7. 最终 clip ==========
    reward = max(-1.0, min(1.0, reward))
    return reward

from typing import Optional
import argparse
import fire
import os
import sys
import pandas as pd
import ai_economist.foundation as foundation
import numpy as np
import matplotlib.pyplot as plt
import yaml
from time import time
from collections import defaultdict
import re
from simulate_utils import *
import pickle as pkl
from itertools import product
from dateutil.relativedelta import relativedelta
GOOD_DECISIONS_DF = None

def load_good_decisions():
    """Lazily load good decision data"""
    global GOOD_DECISIONS_DF
    if GOOD_DECISIONS_DF is None:
        csv_path = '/workspace/QWEN2.5_42_GRPO_700step-/QWEN2.5_42_GRPO_1/data/gpt-3-noperception-reflection-1-100agents-240months/good_decisions.csv'
        if os.path.exists(csv_path):
            GOOD_DECISIONS_DF = pd.read_csv(csv_path)
            print(f"Loaded {len(GOOD_DECISIONS_DF)} good decision examples")
        else:
            print(f"Good decision file not found: {csv_path}")
            GOOD_DECISIONS_DF = pd.DataFrame()
    return GOOD_DECISIONS_DF
with open('config.yaml', "r") as f:
    run_configuration = yaml.safe_load(f)
env_config = run_configuration.get('env')

def get_economic_state(env):
    """Determine current economic state using nominal GDP = Supply × Price (aligned with training data)."""
    # need at least 6 months for 2 quarters
    if env.world.timestep < 6:
        # unemployment from states (if available), growth=0
        states = env.dense_log.get("states", [])
        unemployment_rate = 0.0
        if len(states) > 0:
            m = len(states) - 1
            unemployed, employed = 0, 0
            for agent_id_str, agent_state in states[m].items():
                if (not agent_id_str.isdigit()) or (not isinstance(agent_state, dict)):
                    continue
                job = agent_state.get("endogenous", {}).get("job")
                if job == "Unemployment":
                    unemployed += 1
                else:
                    employed += 1
            labor_force = employed + unemployed
            unemployment_rate = unemployed / labor_force if labor_force > 0 else 0.0
        return "Normal", unemployment_rate, 0.0

    actions = env.dense_log.get("actions", [])
    prices = list(getattr(env.world, "price", []))

    A = 1.0
    H = 168.0

    def month_supply(t):
        if t < 0 or t >= len(actions):
            return 0.0
        month_actions = actions[t] or {}
        total = 0.0
        for agent_id, action in month_actions.items():
            if agent_id == "p":
                continue
            if isinstance(action, (list, tuple)) and len(action) >= 1:
                labor = int(action[0])
            elif isinstance(action, dict):
                labor = int(action.get("SimpleLabor", 0))
            else:
                labor = 0
            total += labor * H * A
        return total

    def month_price(t):
        if t < 0:
            return 1.0
        if t < len(prices):
            return float(prices[t])
        return float(prices[-1]) if prices else 1.0

    def month_gdp(t):
        return month_supply(t) * month_price(t)

    # last 6 months: t-6..t-1 (since timestep is current month index)
    T = env.world.timestep
    q1_months = list(range(T-6, T-3))   # older quarter
    q2_months = list(range(T-3, T))     # latest quarter

    q1_gdp = sum(month_gdp(t) for t in q1_months)
    q2_gdp = sum(month_gdp(t) for t in q2_months)

    q2_growth = (q2_gdp - q1_gdp) / (q1_gdp + 1e-8)

    # unemployment_rate: align with training definition (job == "Unemployment")
    states = env.dense_log.get("states", [])
    if len(states) > 0:
        m = len(states) - 1
        unemployed, employed = 0, 0
        for agent_id_str, agent_state in states[m].items():
            if (not agent_id_str.isdigit()) or (not isinstance(agent_state, dict)):
                continue
            job = agent_state.get("endogenous", {}).get("job")
            if job == "Unemployment":
                unemployed += 1
            else:
                employed += 1
        labor_force = employed + unemployed
        unemployment_rate = unemployed / labor_force if labor_force > 0 else 0.0
    else:
        unemployment_rate = 0.0

    # state classification (keep your thresholds)
    if q2_growth < 0 and (q1_gdp > 0):  # simple recession-ish signal
        return "Recession", unemployment_rate, q2_growth
    elif q2_growth > 0.02:
        return "Boom", unemployment_rate, q2_growth
    else:
        return "Normal", unemployment_rate, q2_growth

def gpt_actions(env, obs, dialog_queue, dialog4ref_queue, gpt_path, gpt_error, total_cost, model_type='gpt', seed=42):
    if not os.path.exists(gpt_path):
        os.makedirs(gpt_path)
    curr_rates = obs['p']['PeriodicBracketTax-curr_rates']
    current_time = world_start_time + relativedelta(months=env.world.timestep)
    current_time = current_time.strftime('%Y.%m')
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        skill = this_agent.state['skill']
        wealth = this_agent.inventory['Coin']
        consumption = this_agent.consumption['Coin']
        interest_rate = env.world.interest_rate[-1]
        price = env.world.price[-1]
        tax_paid = obs['p'][f'p{idx}']['PeriodicBracketTax-tax_paid']
        lump_sum = obs['p'][f'p{idx}']['PeriodicBracketTax-lump_sum']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        name = this_agent.endogenous['name']
        age = this_agent.endogenous['age']
        city = this_agent.endogenous['city']
        job = this_agent.endogenous['job']
        offer = this_agent.endogenous['offer']
        actions = env.dense_log['actions']
        states = env.dense_log['states']

        # ========== Buffer-Stock Ratio Matching (Carroll 1997) ==========
        few_shot_examples = ""
        good_df = load_good_decisions()
        
        if len(good_df) > 0 and env.world.timestep > 0:
            # Step 1: Calculate current agent's buffer ratio
            curr_income = this_agent.income['Coin']
            curr_DPI = curr_income + lump_sum - tax_paid
            cash_on_hand = wealth + curr_DPI
            curr_ratio = cash_on_hand / (curr_DPI + 1e-8)
            
            # Step 2: Compute buffer_ratio for all candidates in good_df
            if 'buffer_ratio' not in good_df.columns:
                if 'curr_dpi' in good_df.columns:
                    good_df['curr_DPI'] = good_df['curr_dpi']
                else:
                    good_df['curr_DPI'] = (
                        good_df['curr_income']
                        + good_df.get('curr_lump', 0)
                        - good_df.get('curr_tax', 0)
                    )
                good_df['cash_on_hand'] = good_df['curr_wealth'] + good_df['curr_DPI']
                good_df['buffer_ratio'] = good_df['cash_on_hand'] / (good_df['curr_DPI'] + 1e-8)
            
            # Step 3: Filter by income scale (special handling for very low income)
            if curr_income <= 1e-6:
                # Unemployed or near-zero income: take bottom 30% income samples
                income_filtered = good_df[good_df['curr_income'] <= good_df['curr_income'].quantile(0.3)].copy()
            else:
                # Normal income: select samples within [0.3x, 3.0x] income range
                income_low = curr_income * 0.3
                income_high = curr_income * 3.0
                income_filtered = good_df[
                    (good_df['curr_income'] >= income_low) &
                    (good_df['curr_income'] <= income_high)
                ].copy()
            
            # Step 4: Match by buffer ratio within income-filtered pool
            if len(income_filtered) >= 3:
                # Primary: ±30% buffer ratio range
                candidates = income_filtered[
                    (income_filtered['buffer_ratio'] >= curr_ratio * 0.7) &
                    (income_filtered['buffer_ratio'] <= curr_ratio * 1.3)
                ].copy()
            else:
                candidates = pd.DataFrame()
            
            # Fallback 1: Relax buffer ratio to ±50%
            if len(candidates) < 3 and len(income_filtered) >= 3:
                candidates = income_filtered[
                    (income_filtered['buffer_ratio'] >= curr_ratio * 0.5) &
                    (income_filtered['buffer_ratio'] <= curr_ratio * 1.5)
                ].copy()
            
            # Fallback 2: Use income-filtered pool only
            if len(candidates) < 3:
                candidates = income_filtered.copy()
            
            # Fallback 3: Global pool sorted by ratio distance
            if len(candidates) < 3:
                good_df['ratio_distance'] = abs(good_df['buffer_ratio'] - curr_ratio)
                candidates = good_df.nsmallest(20, 'ratio_distance')
            
            # Step 5: Select top-3 with employment/unemployment balance (2+1 structure)
            if len(candidates) > 0:
                employed = candidates[candidates['work_decision'] == 1.0]
                unemployed = candidates[candidates['work_decision'] == 0.0]
        
                top_decisions = []
                # Select 2 employed samples by highest score
                if len(employed) >= 2:
                    top_decisions.append(employed.nlargest(2, 'score'))
                # Select 1 unemployed sample by highest score
                if len(unemployed) >= 1:
                    top_decisions.append(unemployed.nlargest(1, 'score'))
                else:
                    # Fallback: if no unemployed samples, take 3 employed
                    top_decisions.append(employed.nlargest(3, 'score'))
                
                top_decisions = pd.concat(top_decisions)
                
                # Step 6: Format few-shot examples for prompt injection
                few_shot_examples = f"\n\n**Examples of good economic decisions (similar income level and buffer ratio ~{curr_ratio:.1f}):**\n"
                for i, (_, row) in enumerate(top_decisions.iterrows(), 1):
                    few_shot_examples += f"Example {i}: Income ${row['curr_income']:.0f}, Wealth ${row['curr_wealth']:.0f}, Buffer Ratio {row['buffer_ratio']:.1f} → Work {row['work_decision']:.0f}, Consumption {row['consumption_prop']:.2f}\n"
        
                few_shot_examples += f"\nYour current situation: Income ${curr_income:.0f}, Wealth ${wealth:.0f}, Buffer Ratio {curr_ratio:.1f}\n"
                few_shot_examples += "Note: Buffer Ratio = (Wealth + DPI) / DPI, where DPI = after-tax income including transfers. Higher ratio suggests more financial cushion.\n"
        # ========== End of Buffer-Stock Ratio Few-Shot Matching ==========
        
        # ========== Inject macroeconomic signals ==========
        macro_signal = ""
        if env.world.timestep > 0:
            economic_state, unemployment_rate, gdp_growth = get_economic_state(env)
            
            macro_signal = f'''
            **Current Economic Indicators:**
            - Overall unemployment rate: {unemployment_rate*100:.1f}%
            - Recent GDP growth: {gdp_growth*100:.1f}% (quarterly)
            - Economic state: {economic_state}
            
            **Economic Context:**
            '''
            if economic_state == "Recession":
                macro_signal += "The economy has experienced negative growth for two consecutive quarters (technical recession). Economists suggest maintaining essential spending to support recovery while being cautious with major purchases."
            elif economic_state == "Boom":
                macro_signal += "The economy is growing strongly. This may be a good time to consider saving for future uncertainties, as economic cycles are normal."
            else:
                macro_signal += "The economy is stable. Balanced consumption and saving strategies are recommended based on your personal financial situation."
        # ========== End of macroeconomic signals ==========
        
        problem_prompt = f'''
            You're {name}, a {age}-year-old individual living in {city}. As with all Americans, a portion of your monthly income is taxed by the federal government. This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings.
            Now it's {current_time}.
        '''

        if job == 'Unemployment':
            job_prompt = f'''
                In the previous month, you became unemployed and had no income. Now, you are invited to work as a(an) {offer} with monthly salary of ${skill*max_l:.2f}.
            '''
        else:
            if skill >= states[-1][str(idx)]['skill']:
                job_prompt = f'''
                    In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is increased compared to the last month due to the inflation of labor market.
                '''
            else:
                job_prompt = f'''
                    In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is decreased compared to the last month due to the deflation of labor market.
                '''
        
        if (consumption <= 0) and (len(actions) > 0) and (actions[-1].get('SimpleConsumption', 0) > 0):
            consumption_prompt = f'''
                Besides, you had no consumption due to shortage of goods.
            '''
        else:
            consumption_prompt = f'''
                Besides, your consumption was ${consumption:.2f}.
            '''
        
        if env._components_dict['PeriodicBracketTax'].tax_model == 'us-federal-single-filer-2018-scaled':
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                In this month, the government sets the brackets: {format_numbers(brackets)} and their corresponding rates: {format_numbers(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        else:
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                In this month, according to the optimal taxation theory, Saez Tax, the brackets are not changed: {format_numbers(brackets)} but the government has updated corresponding rates: {format_percentages(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        
        if env.world.timestep == 0:
            price_prompt = f'''Meanwhile, in the consumption market, the average price of essential goods is now at ${price:.2f}.'''
        else:
            if price >= env.world.price[-2]:
                price_prompt = f'''Meanwhile, inflation has led to a price increase in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
            else:
                price_prompt = f'''Meanwhile, deflation has led to a price decrease in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
        
        job_prompt = prettify_document(job_prompt)
        obs_prompt = f'''
            {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt}
            Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%.{macro_signal}{few_shot_examples}
            With all these factors in play, and considering aspects like your living costs, any future aspirations, the broader economic trends, and how your buffer ratio compares to the examples above, how is your willingness to work this month? Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price?
            **IMPORTANT:** Reply with ONLY a valid JSON object. Required format: {{"work": <number>, "consumption": <number>}} where both are values between 0 and 1 with intervals of 0.02.
        '''
        obs_prompt = prettify_document(obs_prompt)
        dialog_queue[idx].append({'role': 'user', 'content': obs_prompt})
        dialog4ref_queue[idx].append({'role': 'user', 'content': obs_prompt})
        if 'prompts' not in env.dense_log:
            env.dense_log['prompts'] = []
        if len(env.dense_log['prompts']) <= env.world.timestep:
            env.dense_log['prompts'].append({})
        env.dense_log['prompts'][env.world.timestep][str(idx)] = obs_prompt
    
    def action_check(actions):
        if len(actions) != 2:
            return False
        else:
            return (actions[0] >= 0) & (actions[0] <= 1) & (actions[1] >= 0) & (actions[1] <= 1)
    
    if env.world.timestep%3 == 0 and env.world.timestep > 0:
        results, cost = get_multiple_completion([list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:] for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)], model_type=model_type, seed=seed)
        total_cost += cost
    else:
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog_queue], model_type=model_type, seed=seed)
        total_cost += cost
    
    actions = {}
    for idx in range(env.num_agents):
        content = results[idx]
        try:
            extracted_actions = list(eval(content).values())
            if not action_check(extracted_actions):
                extracted_actions = [1, 0.5]
                gpt_error += 1
        except:
            extracted_actions = [1, 0.5]
            gpt_error += 1
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions
        dialog_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
        dialog4ref_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
    
    actions['p'] = [0]
    for idx, agent_dialog in enumerate(dialog_queue):
        with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
            for dialog in list(agent_dialog)[-2:]:
                f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
    
    if (env.world.timestep+1)%3 == 0:
        reflection_prompt = '''Given the previous quarter's economic environment, reflect on the labor, consumption, and financial markets, as well as their dynamics. What conclusions have you drawn?
        Your answer must be less than 200 words!'''
        reflection_prompt = prettify_document(reflection_prompt)
        for idx in range(env.num_agents):
            # dialog_queue[idx].append({'role': 'user', 'content': reflection_prompt})
            dialog4ref_queue[idx].append({'role': 'user', 'content': reflection_prompt})
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog4ref_queue], temperature=0, max_tokens=200, model_type=model_type, seed=seed    )
        total_cost += cost
        for idx in range(env.num_agents):
            content = results[idx]
            # dialog_queue[idx].append({'role': 'assistant', 'content': content})
            dialog4ref_queue[idx].append({'role': 'assistant', 'content': content})
        
        for idx, agent_dialog in enumerate(dialog4ref_queue):
            with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
                for dialog in list(agent_dialog)[-2:]:
                    f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
    
    return actions, gpt_error, total_cost

def complex_actions(env, obs, beta=0.1, gamma=0.1, h=1):

    def consumption_len(price, wealth, curr_income, last_income, interest_rate):
        c = (price/(1e-8+wealth+curr_income))**beta
        c = min(max(c//0.02, 0), 50)
        return c
    def consumption_cats(price, wealth, curr_income, last_income, interest_rate):
        h1 = h / (1 + interest_rate)
        g = curr_income/(last_income+1e-8) - 1
        d = wealth/(last_income+1e-8) - h1
        c = 1 + (d - h1*g)/(1 + g + 1e-8)
        c = min(max(c*curr_income/(wealth+curr_income+1e-8)//0.02, 0), 50)
        return c
    def work_income_wealth(price, wealth, curr_income, last_income, expected_income, interest_rate):
        return int(np.random.uniform() < (curr_income/(wealth*(1 + interest_rate)+1e-8))**gamma)
    
    consumption_funs = [consumption_len, consumption_cats]
    work_funs = [work_income_wealth]

    actions = {}
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        price = env.world.price[-1]
        wealth = this_agent.inventory['Coin']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        max_income = max_l * this_agent.state['skill']
        last_income = this_agent.income['Coin']
        expected_income = max_l * this_agent.state['expected skill']
        interest_rate = env.world.interest_rate[-1]
        if 'consumption_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['consumption_fun_idx'] = np.random.choice(range(len(consumption_funs)))
        if 'work_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['work_fun_idx'] = np.random.choice(range(len(work_funs)))
        work_fun = work_funs[this_agent.endogenous['work_fun_idx']]
        l = work_fun(price, wealth, max_income, last_income, expected_income, interest_rate)
        curr_income = l * max_income
        consumption_fun = consumption_funs[this_agent.endogenous['consumption_fun_idx']]
        c = consumption_fun(price, wealth, curr_income, last_income, interest_rate)
        actions[str(idx)] = [l, c]
    actions['p'] = [0]
    return actions
    

def main(policy_model='gpt', num_agents=100, episode_length=240, dialog_len=3, beta=0.1, gamma=0.1, h=1, max_price_inflation=0.1, max_wage_inflation=0.05, model_type='gpt', seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    env_config['seed'] = seed
    env_config['n_agents'] = num_agents
    env_config['episode_length'] = episode_length
    if policy_model == 'gpt':
        total_cost = 0
        env_config['flatten_masks'] = False
        env_config['flatten_observations'] = False
        env_config['components'][0]['SimpleLabor']['scale_obs'] = False
        env_config['components'][1]['PeriodicBracketTax']['scale_obs'] = False
        env_config['components'][3]['SimpleSaving']['scale_obs'] = False
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation
        
        gpt_error = 0
        from collections import deque
        dialog_queue = [deque(maxlen=dialog_len) for _ in range(env_config['n_agents'])]
        dialog4ref_queue = [deque(maxlen=7) for _ in range(env_config['n_agents'])]

    elif policy_model == 'complex':
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation

    t = time()
    env = foundation.make_env_instance(**env_config)
    obs = env.reset()
    actions = {}
    if policy_model == 'complex':
        policy_model_save = f'{policy_model}-{beta}-{gamma}-{h}-{max_price_inflation}-{max_wage_inflation}'
    if policy_model == 'gpt':
        policy_model_save = f'{policy_model}-{dialog_len}-noperception-reflection-1'
    policy_model_save = f'{policy_model_save}-{num_agents}agents-{episode_length}months'
    if not os.path.exists(f'{save_path}data/{policy_model_save}'):
        os.makedirs(f'{save_path}data/{policy_model_save}')
    if not os.path.exists(f'{save_path}figs/{policy_model_save}'):
        os.makedirs(f'{save_path}figs/{policy_model_save}')
    for epi in range(env.episode_length):
        if policy_model == 'gpt':
            actions, gpt_error, total_cost = gpt_actions(env, obs, dialog_queue, dialog4ref_queue, f'{save_path}data/{policy_model_save}/dialogs', gpt_error, total_cost, model_type=model_type, seed=seed)
        elif policy_model == 'complex':
            actions = complex_actions(env, obs, beta=beta, gamma=gamma, h=h)
        obs, rew, done, info = env.step(actions)
        if (epi+1) % 3 == 0:
            print(f'step {epi+1} done, cost {time()-t:.1f}s')
            if policy_model == 'gpt':
                print(f'#errors: {gpt_error}, cost ${total_cost:.1f} so far')
            t = time()
        if (epi+1) % 6 == 0 or epi+1 == env.episode_length:
            with open(f'{save_path}data/{policy_model_save}/actions_{epi+1}.pkl', 'wb') as f:
                pkl.dump(actions, f)
            with open(f'{save_path}data/{policy_model_save}/obs_{epi+1}.pkl', 'wb') as f:
                pkl.dump(obs, f)
            with open(f'{save_path}data/{policy_model_save}/env_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env, f)
            if policy_model == 'gpt':
                with open(f'{save_path}data/{policy_model_save}/dialog_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog_queue, f)
                with open(f'{save_path}data/{policy_model_save}/dialog4ref_{epi+1}.pkl', 'wb') as f:
                    pkl.dump(dialog4ref_queue, f)
            with open(f'{save_path}data/{policy_model_save}/dense_log_{epi+1}.pkl', 'wb') as f:
                pkl.dump(env.dense_log, f)
                
    with open(f'{save_path}data/{policy_model_save}/dense_log.pkl', 'wb') as f:
        pkl.dump(env.dense_log, f)
        
    if policy_model == 'gpt':
        print(f'#gpt errors: {gpt_error}')

if __name__ == "__main__":
    fire.Fire(main)

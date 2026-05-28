"""
CNE-6 基本面因子计算
换手率、市值、杠杆、估值、分红、财务质量、投资、增长、分析师预测
"""
import numpy as np
import pandas as pd
from datatools import get_basic, get_finance, get_finance_ttm, get_report_roll


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def weight_sum(data, half_life, normalize=True):
    window = len(data)
    w = 0.5**(np.arange(window) / half_life)[::-1]
    if normalize:
        w /= w.sum()
    return np.nansum(w * data)

def calc_slope(x):
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    # 构造权重并计算斜率
    w = np.arange(n) - (n - 1) / 2.0
    return np.dot(x, w) / np.sum(w**2)

def MAD_winsorize(x, multiplier=5):
    x_M = np.nanmedian(x)
    x_MAD = np.nanmedian(np.abs(x-x_M))
    upper = x_M + multiplier * x_MAD
    lower = x_M - multiplier * x_MAD
    x[x>upper] = upper
    x[x<lower] = lower
    return x

def align_to_trade_dates(data, start_date, end_date):
    data = data.set_index(['ts_code', 'ann_date']).reindex(
    pd.MultiIndex.from_product([data['ts_code'].unique(), pd.date_range(data['ann_date'].min(), end_date, freq='D')],
    names=['ts_code', 'ann_date'])).groupby(level='ts_code').ffill().reset_index()
    data = data[data['ann_date'].between(start_date, end_date)]
    trade_cal = pd.to_datetime(pd.read_csv('data/trade_cal.csv')['cal_date'].unique().tolist())
    data = data[data['ann_date'].isin(trade_cal)]

# ── 主函数 ────────────────────────────────────────────────────────────────────

def calc_fund_factors(start_date, end_date, allstocks):
    """
    计算 CNE-6 基本面因子

    Parameters
    ----------
    start_date, end_date : str  'YYYY-MM-DD'
    allstocks : list[str]  股票池

    Returns
    -------
    DataFrame  主键 ts_code + cal_date
    """
    # basic 数据只需1年（换手率最长252日）
    load_start = (pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    # 财务数据需要5年+（增长率计算）
    fin_start = (pd.Timestamp(start_date) - pd.DateOffset(years=7)).strftime('%Y-%m-%d')
    # TTM 数据需要1年
    ttm_start = (pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print('加载基础数据...')
    basic = get_basic(codes=allstocks, start_date=load_start, end_date=end_date,
                      fields=['circ_mv','total_mv' ,'pb', 'pe_ttm', 'dv_ttm', 'turnover_rate'])

    print('加载财务数据...')
    fin = get_finance(codes=allstocks, start_date=fin_start, end_date=end_date, fields=[
        'total_assets', 'total_liab', 'total_ncl','oth_eqt_tools_p_shr',
        'total_hldr_eqy_inc_min_int','c_cash_equ_end_period']).ffill()
    fin.iloc[:, 3:] = fin.iloc[:, 3:]/10000  # 财务数据单位转换为万元

    print('加载TTM数据...')
    fin_ttm = get_finance_ttm(codes=allstocks, start_date=ttm_start, end_date=end_date, fields=[
        'total_revenue_ttm', 'n_income_attr_p_ttm', 'total_cogs_ttm', 'ebit_ttm', 
        'depr_fa_coga_dpba_ttm', 'amort_intang_assets_ttm', 'lt_amort_deferred_exp_ttm',
        'n_cashflow_act_ttm', 'n_cashflow_inv_act_ttm'])
    fin_ttm.iloc[:, 3:] = fin_ttm.iloc[:, 3:]/10000  # TTM数据单位转换为万元

    # ── 构建基础矩阵 ─────────────────────────────────────────────────────────
    dates = np.sort(basic['trade_date'].unique())
    universe = np.sort(basic['ts_code'].unique())
    
    n_dates, n_stocks = basic.shape
    print(f'矩阵: {n_dates} 交易日 × {n_stocks} 只股票')

    # basic 数据 pivot
    circ_mv = basic.pivot(index='trade_date', columns='ts_code', values='circ_mv') \
                    .reindex(index=dates, columns=universe)  # 单位：万元
    total_mv = basic.pivot(index='trade_date', columns='ts_code', values='total_mv') \
                    .reindex(index=dates, columns=universe)  # 单位：万元
    pb = basic.pivot(index='trade_date', columns='ts_code', values='pb') \
              .reindex(index=dates, columns=universe)
    pe_ttm = basic.pivot(index='trade_date', columns='ts_code', values='pe_ttm') \
                   .reindex(index=dates, columns=universe)
    dv_ttm = basic.pivot(index='trade_date', columns='ts_code', values='dv_ttm') \
                   .reindex(index=dates, columns=universe)
    turnover = basic.pivot(index='trade_date', columns='ts_code', values='turnover_rate') \
                     .reindex(index=dates, columns=universe)
    
    total_revenue_ttm = fin_ttm.pivot(index='trade_date', columns='ts_code', values='total_revenue_ttm').reindex(index=dates, columns=universe)
    n_income_attr_p_ttm = fin_ttm.pivot(index='trade_date', columns='ts_code', values='n_income_attr_p_ttm').reindex(index=dates, columns=universe)
    total_cogs_ttm = fin_ttm.pivot(index='trade_date', columns='ts_code', values='total_cogs_ttm').reindex(index=dates, columns=universe)
    ebit_ttm = fin_ttm.pivot(index='trade_date', columns='ts_code', values='ebit_ttm').reindex(index=dates, columns=universe)
    depr_fa_coga_dpba_ttm = fin_ttm.pivot(index='trade_date', columns='ts_code', values='depr_fa_coga_dpba_ttm').reindex(index=dates, columns=universe)
    amort_intang_assets_ttm = fin_ttm.pivot(index='trade_date', columns='ts_code', values='amort_intang_assets_ttm').reindex(index=dates, columns=universe)
    lt_amort_deferred_exp_ttm = fin_ttm.pivot(index='trade_date', columns='ts_code', values='lt_amort_deferred_exp_ttm').reindex(index=dates, columns=universe)
    n_cashflow_act_ttm = fin_ttm.pivot(index='trade_date', columns='ts_code', values='n_cashflow_act_ttm').reindex(index=dates, columns=universe)
    n_cashflow_inv_act_ttm = fin_ttm.pivot(index='trade_date', columns='ts_code', values='n_cashflow_inv_act_ttm').reindex(index=dates, columns=universe)

    total_assets = fin.pivot(index='trade_date', columns='ts_code', values='total_assets').reindex(index=dates, columns=universe)
    total_liab = fin.pivot(index='trade_date', columns='ts_code', values='total_liab').reindex(index=dates, columns=universe)
    total_ncl = fin.pivot(index='trade_date', columns='ts_code', values='total_ncl').reindex(index=dates, columns=universe)
    total_hldr_eqy = fin.pivot(index='trade_date', columns='ts_code', values='total_hldr_eqy_inc_min_int').reindex(index=dates, columns=universe)
    oth_eqt_tools_p_shr = fin.pivot(index='trade_date', columns='ts_code', values='oth_eqt_tools_p_shr').reindex(index=dates, columns=universe)
    
    # ================================================================
    # 换手率因子
    # ================================================================
    print('计算换手率因子...')
    stom = np.log(turnover.rolling(21, min_periods=1).sum()).loc[start_date:end_date]
    
    exp_STOM_sum = 0
    for i in range(3):
        exp_STOM_sum += np.exp(stom.shift(i * 21))
    stoq = np.log(exp_STOM_sum / 3).loc[start_date:end_date]

    exp_STOM_sum = 0
    for i in range(12):
        exp_STOM_sum += np.exp(stom.shift(i * 21))
    stoa = np.log(exp_STOM_sum / 12).loc[start_date:end_date]

    atvr = turnover.rolling(window=252, min_periods=1).apply(
        lambda x: weight_sum(x, 63, normalize=False), raw=True).loc[start_date:end_date]


    # ================================================================
    # 市值因子
    # ================================================================
    print('计算市值因子...')
    lncap = np.log(circ_mv).loc[start_date:end_date]
    midcap = (lncap ** 3)
    midcap = midcap.loc[start_date:end_date]
    # ================================================================
    # 估值/杠杆因子
    # ================================================================
    print('计算估值/杠杆因子...')

    mlev = (total_mv + oth_eqt_tools_p_shr + total_ncl) / total_mv
    mlev = mlev.loc[start_date:end_date]
    blev = (total_hldr_eqy + oth_eqt_tools_p_shr + total_ncl) / total_mv
    blev = blev.loc[start_date:end_date]
    dtoa = total_liab / total_assets
    dtoa = dtoa.loc[start_date:end_date]

    btp = 1.0 / pb
    btp = btp.loc[start_date:end_date]
    etp = 1.0 / pe_ttm
    etp = etp.loc[start_date:end_date]
    cftp = n_income_attr_p_ttm  / total_mv
    cftp = cftp.loc[start_date:end_date]
    ebit_ev = ebit_ttm / total_mv
    ebit_ev = ebit_ev.loc[start_date:end_date]

    dp = dv_ttm.loc[start_date:end_date]
    
    # ================================================================
    # 盈利能力
    # ================================================================
    print('计算盈利能力...')
    ato = total_revenue_ttm / total_assets
    gp = (total_revenue_ttm - total_cogs_ttm) / total_assets
    gpm = (total_revenue_ttm - total_cogs_ttm) / total_revenue_ttm
    roa = n_income_attr_p_ttm / total_assets
    
    # ================================================================
    # 财务质量因子（5年波动率）
    # ================================================================
    print('计算财务质量因子...')
    var_ttm = get_finance_ttm(codes=allstocks, start_date=ttm_start, end_date=end_date, fields=[
        'total_revenue_ttm', 'n_income_attr_p_ttm','n_incr_cash_cash_equ_ttm'], align_trade_date=False)
    var_ttm = var_ttm[var_ttm.stat_date.str.contains('1231')]
    var_ttm.iloc[:, 3:] = var_ttm.iloc[:, 3:]/10000
    
    means = var_ttm.set_index('trade_date').groupby('ts_code').rolling(5, min_periods=1)[['total_revenue_ttm','n_income_attr_p_ttm','n_incr_cash_cash_equ_ttm']].mean().reset_index()
    means = align_to_trade_dates(means, start_date, end_date)
    stds = var_ttm.set_index('trade_date').groupby('ts_code').rolling(5, min_periods=1)[['total_revenue_ttm','n_income_attr_p_ttm','n_incr_cash_cash_equ_ttm']].std().reset_index()
    stds = align_to_trade_dates(stds, start_date, end_date)
    
    var_sales = stds.pivot(index='trade_date', columns='ts_code', values='total_revenue_ttm').reindex(index=dates, columns=universe) / \
                means.pivot(index='trade_date', columns='ts_code', values='total_revenue_ttm').reindex(index=dates, columns=universe).loc[start_date:end_date]
    var_earnings = stds.pivot(index='trade_date', columns='ts_code', values='n_income_attr_p_ttm').reindex(index=dates, columns=universe) / \
                   means.pivot(index='trade_date', columns='ts_code', values='n_income_attr_p_ttm').reindex(index=dates, columns=universe).loc[start_date:end_date]
    var_cashflows = stds.pivot(index='trade_date', columns='ts_code', values='n_incr_cash_cash_equ_ttm').reindex(index=dates, columns=universe) / \
                    means.pivot(index='trade_date', columns='ts_code', values='n_incr_cash_cash_equ_ttm').reindex(index=dates, columns=universe).loc[start_date:end_date]

    # ================================================================
    # 应计项目
    # ================================================================
    print('计算应计项目...')
    noa = get_finance(codes=allstocks, start_date='2022-11-01', end_date='2025-12-31', fields=[
        'total_assets', 'total_liab','non_cur_liab_due_1y','c_cash_equ_end_period', 'total_ncl', 'st_borr'], align_trade_date=False).ffill()
    noa.iloc[:, 3:] = noa.iloc[:, 3:]/10000
    noa['noa'] = noa.eval('total_assets - c_cash_equ_end_period - total_liab + non_cur_liab_due_1y + total_ncl + st_borr')
    noa['noa'] = noa.groupby('ts_code')['noa'].diff()
    noa = align_to_trade_dates(noa, start_date, end_date).pivot(index='trade_date', columns='ts_code', values='noa').reindex(index=dates, columns=universe)
    DA = depr_fa_coga_dpba_ttm + amort_intang_assets_ttm + lt_amort_deferred_exp_ttm
    accr_bs = (DA - noa) / total_assets
    accr_bs = accr_bs.loc[start_date:end_date]

    accr_cf = (n_cashflow_act_ttm + n_cashflow_inv_act_ttm -n_income_attr_p_ttm - DA) / total_assets
    accr_cf = accr_cf.loc[start_date:end_date]


    # ================================================================
    # 投资质量（5年增长率）
    # ================================================================
    print('计算投资质量...')
    growth = get_finance(codes=allstocks, start_date=ttm_start, end_date=end_date, fields=[
        'total_assets', 'total_share'], align_trade_date=False).ffill()
    growth = growth[growth.stat_date.str.contains('1231')]
    growth.iloc[:, 3:] = growth.iloc[:, 3:]/10000

    ta_growth = growth.set_index('trade_date').groupby('ts_code')['total_assets'].rolling(5, min_periods=1).apply(calc_slope, raw=True).reset_index()
    ta_growth = align_to_trade_dates(ta_growth, start_date, end_date).pivot(index='trade_date', columns='ts_code', values='total_assets').reindex(index=dates, columns=universe).loc[start_date:end_date]
    
    issuance_growth = growth.set_index('trade_date').groupby('ts_code')['total_share'].rolling(5, min_periods=1).apply(calc_slope, raw=True).reset_index()
    issuance_growth = align_to_trade_dates(issuance_growth, start_date, end_date).pivot(index='trade_date', columns='ts_code', values='total_share').reindex(index=dates, columns=universe).loc[start_date:end_date]
        
    growth = get_finance_ttm(codes=allstocks, start_date=ttm_start, end_date=end_date, fields=[
    'c_pay_acq_const_fiolta_ttm', 'total_revenue_ttm', 'n_income_attr_p_ttm','basic_eps_ttm'], align_trade_date=False).ffill()
    growth = growth[growth.stat_date.str.contains('1231')]
    growth.iloc[:, 3:] = growth.iloc[:, 3:]/10000

    capex_growth = growth.set_index('trade_date').groupby('ts_code')['c_pay_acq_const_fiolta_ttm'].rolling(5, min_periods=1).apply(calc_slope, raw=True).reset_index()
    capex_growth = align_to_trade_dates(capex_growth, start_date, end_date).pivot(index='trade_date', columns='ts_code', values='c_pay_acq_const_fiolta_ttm').reindex(index=dates, columns=universe).loc[start_date:end_date]
    
    eps_growth = growth.set_index('trade_date').groupby('ts_code')['basic_eps_ttm'].rolling(5, min_periods=1).apply(calc_slope, raw=True).reset_index()
    eps_growth = align_to_trade_dates(eps_growth, start_date, end_date).pivot(index='trade_date', columns='ts_code', values='basic_eps_ttm').reindex(index=dates, columns=universe).loc[start_date:end_date]
    
    growth['sps_ttm'] = growth.eval('total_revenue_ttm / n_income_attr_p_ttm * basic_eps_ttm')
    sps_growth = growth.set_index('trade_date').groupby('ts_code')['sps_ttm'].rolling(5, min_periods=1).apply(calc_slope, raw=True).reset_index()
    sps_growth = align_to_trade_dates(sps_growth, start_date, end_date).pivot(index='trade_date', columns='ts_code', values='sps_ttm').reindex(index=dates, columns=universe).loc[start_date:end_date]

    # ================================================================
    # 分析师预测因子
    # ================================================================
    
    print('加载分析师预测数据...')
    years = list(range(pd.to_datetime(start_date).year, pd.to_datetime(end_date).year + 5))
    pred = get_report_roll(codes=allstocks, year=years, start_date=load_start, end_date=end_date, fields=[
        'np_roll_std', 'np_roll_mean', 'rd_roll_mean', 'roe_roll_mean', 'eps_roll_mean', 'roll_cnt'])
    pred = pred[pred['quarter'].notna()]
    
    print('计算分析师预测因子...')
    exact_year = pred[pred['trade_date'].dt.year ==pred['quarter'].apply(lambda x: int(x[:4]))]
    np_std = exact_year.pivot(index='trade_date', columns='ts_code', values='np_roll_std') \
                   .reindex(index=dates, columns=universe)
    sdafep = np_std / total_mv
    sdafep = sdafep.loc[start_date:end_date]
    np_mean = exact_year.pivot(index='trade_date', columns='ts_code', values='np_roll_mean') \
                   .reindex(index=dates, columns=universe)
    apebs = np_mean / total_mv
    delta_apebs = sum([apebs.pct_change(periods=63).shift(i*63).div(i+1) for i in range(4)]).loc[start_date:end_date]
    apebs = apebs.loc[start_date:end_date]
    adtp = exact_year.pivot(index='trade_date', columns='ts_code', values='rd_roll_mean') \
                   .reindex(index=dates, columns=universe)
    adtp = adtp.loc[start_date:end_date]

    # 预测4年增长率
    future = pred[pred['trade_date'].dt.year ==pred['quarter'].apply(lambda x: int(x[:4]+4))]  # 4年后预测值近似未来3年增长率
    pred_growth3 = future.pivot(index='trade_date', columns='ts_code', values='roe_roll_mean') \
                  .reindex(index=dates, columns=universe)
    pred_growth3 = pred_growth3.loc[start_date:end_date]
    # 变化比率
    eps_mean = exact_year.pivot(index='trade_date', columns='ts_code', values='eps_roll_mean') \
                   .reindex(index=dates, columns=universe)
    delta_eps = sum([eps_mean.pct_change(periods=63).shift(i*63).div(i+1) for i in range(4)]).loc[start_date:end_date]

    roll_cnt = exact_year.pivot(index='trade_date', columns='ts_code', values='roll_cnt') \
                   .reindex(index=dates, columns=universe)
    delta_cnt = sum([roll_cnt.pct_change(periods=63).shift(i*63).div(i+1) for i in range(4)]).loc[start_date:end_date]
    # ================================================================
    # 组装输出
    # ================================================================
    print('组装结果...')
    rows = np.repeat(dates, n_stocks)
    cols = np.tile(universe, n_dates)

    result = pd.DataFrame({
        'ts_code': cols, 'cal_date': rows,
        # 换手率
        'STOM': stom.ravel(), 'STOQ': stoq.ravel(), 'STOA': stoa.ravel(),
        'ATVR': atvr.ravel(),
        # 市值
        'LNCAP': lncap.ravel(), 'MIDCAP': midcap.ravel(),
        # 杠杆
        'MLEV': mlev.ravel(), 'BLEV': blev.ravel(), 'DTOA': dtoa.ravel(),
        # 估值
        'BTP': btp.ravel(), 'ETP': etp.ravel(), 'CFTP': cftp.ravel(),
        'EBIT_EV': ebit_ev.ravel(), 'DP': dp.ravel(),
        # 财务质量
        'VAR_SALES': var_sales.ravel(), 'VAR_EARNINGS': var_earnings.ravel(),
        'VAR_CASHFLOWS': var_cashflows.ravel(),
        'ACCR_BS': accr_bs.ravel(), 'ACCR_CF': accr_cf.ravel(),
        # 盈利能力
        'ATO': ato.ravel(), 'GP': gp.ravel(), 'GPM': gpm.ravel(), 'ROA': roa.ravel(),
        # 投资质量
        'TA_GROWTH': ta_growth.ravel(), 'ISSUANCE_GROWTH': issuance_growth.ravel(),
        'CAPEX_GROWTH': capex_growth.ravel(),
        'EPS_GROWTH': eps_growth.ravel(), 'SPS_GROWTH': sps_growth.ravel(),
        # 分析师预测
        'SDAFEP': sdafep.ravel(), 'APEBS': apebs.ravel(),
        'PRED_GROWTH3': pred_growth3.ravel(),
        'ADTP': adtp.ravel(), 'DELTA_APEBS': delta_apebs.ravel(),
        'DELTA_EPS': delta_eps.ravel(), 'DELTA_CNT': delta_cnt.ravel()
    })

    # 截取目标日期范围
    result = result[result['cal_date'].between(start_date, end_date)].copy()
    result = result.sort_values(['ts_code', 'cal_date']).reset_index(drop=True)
    print('完成')
    return result


# ── 使用示例 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    allstocks = pd.read_csv('data/allstock.csv')
    allstocks = allstocks[(allstocks['list_date'] < 20250101) & ~allstocks['ts_code'].str.contains('BJ')].ts_code.tolist()

    df = calc_fund_factors(start_date='2025-01-01', end_date='2025-03-31', allstocks=allstocks)

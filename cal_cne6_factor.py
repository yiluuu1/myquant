import numpy as np
import pandas as pd
from datatools import get_price, get_index_K, get_trade_cal, get_basic, get_finance, get_finance_ttm, get_report_roll

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

def align_to_trade_dates(data, start_date, end_date):
    data = data.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
    data = data.set_index(['ts_code', 'trade_date']).reindex(
    pd.MultiIndex.from_product([data['ts_code'].unique(), pd.date_range(data['trade_date'].min(), end_date, freq='D')],
    names=['ts_code', 'trade_date'])).groupby(level='ts_code').ffill().reset_index()
    data = data[data['trade_date'].between(start_date, end_date)]
    trade_cal = pd.to_datetime(pd.read_csv('data/trade_cal.csv')['cal_date'].unique().tolist())
    data = data[data['trade_date'].isin(trade_cal)]
    return data
    
# ── 主函数 ────────────────────────────────────────────────────────────────────

def calc_cne6_factors(start_date, end_date, allstocks):
    """
    计算 CNE-6 价格类因子

    Parameters
    ----------
    start_date, end_date : str  'YYYY-MM-DD'
    allstocks : list[str]  股票池

    Returns
    -------
    DataFrame  主键 ts_code + cal_date
    """
    # ── 提前加载足够历史数据（覆盖最长窗口 1040 日 + 季节性 5 年月度）──
    load_start = (pd.Timestamp(start_date) - pd.DateOffset(months=13)).strftime('%Y-%m-%d')

    print('加载价格数据...')
    price = get_price(codes=allstocks, start_date=load_start, end_date=end_date, fields=['close', 'pre_close'])

    print('加载指数数据...')
    hs300 = get_index_K(codes=['000300.SH'], start_date=load_start, end_date=end_date, fields=['close', 'pre_close'])

    # ── 构建矩阵 ──────────────────────────────────────────────────────────────
    dates = np.sort(price['trade_date'].unique())
    universe = np.sort(price['ts_code'].unique())

    close_pivot = price.pivot(index='trade_date', columns='ts_code', values='close').reindex(index=dates, columns=universe)
    pre_pivot = price.pivot(index='trade_date', columns='ts_code', values='pre_close').reindex(index=dates, columns=universe)

    ret_mat = np.log(close_pivot/pre_pivot)          # (T, N)

    hs300_close = hs300.set_index('trade_date')['close'].reindex(dates)
    hs300_pre   = hs300.set_index('trade_date')['pre_close'].reindex(dates)
    ret_mkt = np.log(hs300_close / hs300_pre)          # (T,)

    n_dates, n_stocks = ret_mat.shape
    print(f'矩阵: {n_dates} 交易日 × {n_stocks} 只股票')

    # ================================================================
    # 1. Daily std — 252 日窗口，半衰期 42 日
    # ================================================================
    print('计算 Daily std...')
    daily_std = ret_mat.ewm(halflife=42).std().loc[start_date:end_date]

    # ================================================================
    # 2. Cumulative range — Z(T) 为过去 T 个月累积对数收益，T=1..12
    # ================================================================
    print('计算 Cumulative range...')

    rolling_cum = ret_mat.expanding().sum()
    # 方法一
    # result = {}
    # for date in get_trade_cal(start_date=start_date, end_date=end_date):
    #     today = rolling_cum.index.get_loc(date)
    #     values = rolling_cum.iloc[today] - rolling_cum.iloc[[today - i * 21 for i in range(1,13)]]
    #     result[date] = values.max() - values.min()
    # cum_range = pd.DataFrame(result).T.sort_index()

    ### 方法二
    idx_past = np.arange(len(ret_mat))[np.newaxis, :] - (np.arange(1, 13) * 21)[:, np.newaxis]
    arr = rolling_cum.values
    arr_past = np.where(idx_past[:, :, np.newaxis] >= 0, arr[idx_past], np.nan)
    arr_today = arr[np.newaxis, :, :]
    window_rets = arr_today - arr_past
    factor_values = np.nanmax(window_rets, axis=0) - np.nanmin(window_rets, axis=0)
    cum_range = pd.DataFrame(factor_values, index=ret_mat.index, columns=ret_mat.columns).loc[start_date:end_date]

    # ================================================================
    # 3. Short Term reversal — 21 日窗口，半衰期 5 日
    # ================================================================
    print('计算 Short Term reversal...')
    st_rev = ret_mat.rolling(window=21, min_periods=1).mean()
    st_rev = st_rev.rolling(window=21, min_periods=1).apply(lambda x: weight_sum(x, 5, normalize=False), raw=True).loc[start_date:end_date]

    # ================================================================
    # 4. Seasonality — 过去 5 年同月收益率均值
    # ================================================================
    print('计算 Seasonality...')
    season = {}
    for date in get_trade_cal(start_date=start_date, end_date=end_date):
        r_y = []
        for i in range(1, 6):
            his = ret_mat.index.get_indexer([date-pd.Timedelta(days=365*i)], method='ffill')[0]
            r_y.append(ret_mat.iloc[his: his+21].sum())
        season[date] = pd.concat(r_y, axis=1).mean(axis=1)
    season = pd.DataFrame(season).T

    # ================================================================
    # 5. Industry Momentum（暂不做行业中性化）
    # ================================================================
    print('计算 Industry Momentum...')
    #indmom_mat = ret_mat.rolling(window=126, min_periods=1).apply(lambda x: weight_sum(x, 21, normalize=False), raw=True).loc[start_date:end_date]
    indmom_mat = ret_mat.ewm(halflife=21, adjust=False).sum().loc[start_date:end_date]

    # ================================================================
    # 6. BETA + Hist sigma + Historical alpha（同一 CAPM 回归，252日 / 63日）
    # ================================================================
    print('计算 BETA / Hist sigma / Historical alpha...')
    r_mkt2 = pd.DataFrame((ret_mkt ** 2).values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index,columns=ret_mat.columns)
    r_mkt_r = ret_mat*ret_mkt.values[:, None]
    r_mkt_N = pd.DataFrame(ret_mkt.values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index, columns=ret_mat.columns)

    # ---------- 2. 计算各项指数加权和 ----------
    mean_wm2 = r_mkt2.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 63), raw=True)
    mean_wmr = r_mkt_r.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 63), raw=True)
    mean_wm = r_mkt_N.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 63), raw=True)
    mean_wr = ret_mat.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 63), raw=True)

    # ---------- 3. 计算 Beta & Alpha ----------
    var_m = mean_wm2 - mean_wm ** 2
    cov_mr = mean_wmr- mean_wm * mean_wr

    beta = cov_mr / var_m
    beta = beta.loc[start_date:end_date]
    alpha = mean_wr - beta * mean_wm
    alpha = alpha.loc[start_date:end_date]

    var_S = ret_mat.rolling(252, min_periods=1).var()
    var_M = r_mkt_N.rolling(252, min_periods=1).var()
    cov_SM = ret_mat.rolling(252, min_periods=1).cov(ret_mkt)

    # 2. 利用方差分解公式：Var(S - A - B*M) = Var(S - B*M)
    res_var = var_S + beta**2 * var_M - 2 * beta * cov_SM
    hist_sigma = np.sqrt(res_var.clip(lower=0)).loc[start_date:end_date]

    # ================================================================
    # 7. Relative strength — 252 日 / 126 日，11 日窗口均值
    # ================================================================
    print('计算 Relative strength...')
    rs = ret_mat.rolling(window=252, min_periods=1).apply(lambda x: weight_sum(x, 126, normalize=False), raw=True)
    rs = rs.rolling(window=11, min_periods=1).mean().loc[start_date:end_date]

    # ================================================================
    # 8. Long term relative strength — 1040 日 / 260 日，滞后 273 日，11 日均值，取负
    # ================================================================
    long_start = (pd.Timestamp(start_date) - pd.DateOffset(year=5)).strftime('%Y-%m-%d')
    print('加载长期价格数据...')
    price = get_price(codes=allstocks, start_date=long_start, end_date=end_date, fields=['close', 'pre_close'])

    print('加载长期指数数据...')
    hs300 = get_index_K(codes=['000300.SH'], start_date=long_start, end_date=end_date, fields=['close', 'pre_close'])

    # ── 构建矩阵 ──────────────────────────────────────────────────────────────
    close_pivot = price.pivot(index='trade_date', columns='ts_code', values='close').reindex(index=dates, columns=universe)
    pre_pivot = price.pivot(index='trade_date', columns='ts_code', values='pre_close').reindex(index=dates, columns=universe)
    ret_mat = np.log(close_pivot/pre_pivot)          # (T, N)

    hs300_close = hs300.set_index('trade_date')['close'].reindex(dates)
    hs300_pre   = hs300.set_index('trade_date')['pre_close'].reindex(dates)
    ret_mkt = np.log(hs300_close / hs300_pre)    

    print('计算 Long term relative strength...')
    long_term_rs = ret_mat.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260, normalize=False), raw=True)
    long_term_rs = - long_term_rs.rolling(window=11, min_periods=1).mean().loc[start_date:end_date]

    # ================================================================
    # 9. Long term historical alpha — 1040 日 / 260 日，滞后 273 日，11 日均值，取负
    # ================================================================
    print('计算 Long term historical alpha...')
    r_mkt2 = pd.DataFrame((ret_mkt ** 2).values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index,columns=ret_mat.columns)
    r_mkt_r = ret_mat*ret_mkt.values[:, None]
    r_mkt_N = pd.DataFrame(ret_mkt.values[:, None] * np.ones((1, ret_mat.shape[1])),index=ret_mat.index, columns=ret_mat.columns)

    mean_wm2 = r_mkt2.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260), raw=True)
    mean_wmr = r_mkt_r.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260), raw=True)
    mean_wm = r_mkt_N.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260), raw=True)
    mean_wr = ret_mat.rolling(window=1040, min_periods=1).apply(lambda x: weight_sum(x, 260), raw=True)

    var_m = mean_wm2 - mean_wm ** 2
    cov_mr = mean_wmr- mean_wm * mean_wr

    long_term_alpha = mean_wr - cov_mr / var_m * mean_wm
    long_term_alpha = - long_term_alpha.rolling(window=11, min_periods=1).mean().loc[start_date:end_date]

    
    # ================================================================
    # 财务类
    # ================================================================
    
    load_start = (pd.Timestamp(start_date) - pd.DateOffset(months=13)).strftime('%Y-%m-%d')
    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print('加载基础数据...')
    basic = get_basic(codes=allstocks, start_date=load_start, end_date=end_date,
                        fields=['circ_mv','total_mv' ,'pb', 'pe_ttm', 'dv_ttm', 'turnover_rate'])

    print('加载财务数据...')
    fin = get_finance(codes=allstocks, start_date=load_start, end_date=end_date, fields=[
        'total_assets', 'total_liab', 'total_ncl','oth_eqt_tools_p_shr',
        'total_hldr_eqy_exc_min_int','c_cash_equ_end_period']).fillna(0)
    fin.iloc[:, 3:] = fin.iloc[:, 3:]/10000  # 财务数据单位转换为万元

    print('加载TTM数据...')
    fin_ttm = get_finance_ttm(codes=allstocks, start_date=load_start, end_date=end_date, fields=[
        'total_revenue_ttm', 'n_income_attr_p_ttm', 'total_cogs_ttm', 'ebit_ttm', 
        'depr_fa_coga_dpba_ttm', 'amort_intang_assets_ttm', 'lt_amort_deferred_exp_ttm',
        'n_cashflow_act_ttm', 'n_cashflow_inv_act_ttm']).set_index('ts_code').groupby(level=0).ffill().reset_index()
    fin_ttm.iloc[:, 3:] = fin_ttm.iloc[:, 3:]/10000  # TTM数据单位转换为万元

    # ── 构建基础矩阵 ─────────────────────────────────────────────────────────
    dates = np.sort(basic['trade_date'].unique())
    universe = np.sort(basic['ts_code'].unique())

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
    total_hldr_eqy = fin.pivot(index='trade_date', columns='ts_code', values='total_hldr_eqy_exc_min_int').reindex(index=dates, columns=universe)
    oth_eqt_tools_p_shr = fin.pivot(index='trade_date', columns='ts_code', values='oth_eqt_tools_p_shr').reindex(index=dates, columns=universe)   
    # ================================================================
    # 换手率因子
    # ================================================================
    print('计算换手率因子...')
    stom = np.log(turnover.rolling(21, min_periods=1).sum()).loc[start_date:end_date]
    stoq = np.log(turnover.rolling(63, min_periods=1).sum()/3).loc[start_date:end_date]
    stoa = np.log(turnover.rolling(252, min_periods=1).sum()/12).loc[start_date:end_date]
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
    ato = ato.loc[start_date:end_date]
    gp = (total_revenue_ttm - total_cogs_ttm) / total_assets
    gp = gp.loc[start_date:end_date]
    gpm = (total_revenue_ttm - total_cogs_ttm) / total_revenue_ttm
    gpm = gpm.loc[start_date:end_date]
    roa = n_income_attr_p_ttm / total_assets
    roa = roa.loc[start_date:end_date]
    
    # ================================================================
    # 财务质量因子（5年波动率）
    # ================================================================
    print('计算财务质量因子...')
    ttm_start = (pd.Timestamp(start_date) - pd.DateOffset(years=7)).strftime('%Y-%m-%d')
    var_ttm = get_finance_ttm(codes=allstocks, start_date=ttm_start, end_date=end_date, fields=[
        'total_revenue_ttm', 'n_income_attr_p_ttm','n_incr_cash_cash_equ_ttm'], align_trade_date=False).set_index('ts_code').groupby(level=0).ffill().reset_index()
    var_ttm = var_ttm[var_ttm.stat_date.str.contains('1231')]
    var_ttm.iloc[:, 3:] = var_ttm.iloc[:, 3:]/10000

    means = var_ttm.set_index('trade_date').groupby('ts_code').rolling(5, min_periods=1)[['total_revenue_ttm','n_income_attr_p_ttm','n_incr_cash_cash_equ_ttm']].mean().reset_index()
    means = align_to_trade_dates(means, start_date, end_date)
    stds = var_ttm.set_index('trade_date').groupby('ts_code').rolling(5, min_periods=1)[['total_revenue_ttm','n_income_attr_p_ttm','n_incr_cash_cash_equ_ttm']].std().reset_index()
    stds = align_to_trade_dates(stds, start_date, end_date)

    var_sales = stds.pivot(index='trade_date', columns='ts_code', values='total_revenue_ttm').reindex(index=dates, columns=universe) / \
                means.pivot(index='trade_date', columns='ts_code', values='total_revenue_ttm').reindex(index=dates, columns=universe)
    var_sales = var_sales.loc[start_date:end_date]
    var_earnings = stds.pivot(index='trade_date', columns='ts_code', values='n_income_attr_p_ttm').reindex(index=dates, columns=universe) / \
                    means.pivot(index='trade_date', columns='ts_code', values='n_income_attr_p_ttm').reindex(index=dates, columns=universe)
    var_earnings = var_earnings.loc[start_date:end_date]
    var_cashflows = stds.pivot(index='trade_date', columns='ts_code', values='n_incr_cash_cash_equ_ttm').reindex(index=dates, columns=universe) / \
                    means.pivot(index='trade_date', columns='ts_code', values='n_incr_cash_cash_equ_ttm').reindex(index=dates, columns=universe)
    var_cashflows = var_cashflows.loc[start_date:end_date]

    # ================================================================
    # 应计项目
    # ================================================================
    print('计算应计项目...')
    n_1_start = (pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    noa = get_finance(codes=allstocks, start_date=n_1_start, end_date=end_date, fields=[
        'total_assets', 'total_liab','non_cur_liab_due_1y','c_cash_equ_end_period', 'total_ncl', 'st_borr'], align_trade_date=False).fillna(0)
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
        'total_assets', 'total_share'], align_trade_date=False).fillna(0)
    growth = growth[growth.stat_date.str.contains('1231')]
    growth.iloc[:, 3:] = growth.iloc[:, 3:]/10000

    ta_growth = growth.set_index('trade_date').groupby('ts_code')['total_assets'].rolling(5, min_periods=1).apply(calc_slope, raw=True).reset_index()
    ta_growth = align_to_trade_dates(ta_growth, start_date, end_date).pivot(index='trade_date', columns='ts_code', values='total_assets').reindex(index=dates, columns=universe).loc[start_date:end_date]

    issuance_growth = growth.set_index('trade_date').groupby('ts_code')['total_share'].rolling(5, min_periods=1).apply(calc_slope, raw=True).reset_index()
    issuance_growth = align_to_trade_dates(issuance_growth, start_date, end_date).pivot(index='trade_date', columns='ts_code', values='total_share').reindex(index=dates, columns=universe).loc[start_date:end_date]

    growth = get_finance_ttm(codes=allstocks, start_date=ttm_start, end_date=end_date, fields=[
    'c_pay_acq_const_fiolta_ttm', 'total_revenue_ttm', 'n_income_attr_p_ttm','basic_eps_ttm'], align_trade_date=False).set_index('ts_code').groupby(level=0).ffill().reset_index()
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
    delta_apebs = sum([apebs.pct_change(periods=63).fillna(0).shift(i*63).div(i+1) for i in range(4)]).loc[start_date:end_date]
    apebs = apebs.loc[start_date:end_date]
    adtp = exact_year.pivot(index='trade_date', columns='ts_code', values='rd_roll_mean') \
                    .reindex(index=dates, columns=universe)
    adtp = adtp.loc[start_date:end_date]

    # 预测4年增长率
    future = pred[pred['trade_date'].dt.year ==pred['quarter'].apply(lambda x: int(x[:4])+4)]  # 4年后预测值近似未来3年增长率
    pred_growth3 = future.pivot(index='trade_date', columns='ts_code', values='roe_roll_mean') \
                    .reindex(index=dates, columns=universe)
    pred_growth3 = pred_growth3.loc[start_date:end_date]
    # 变化比率
    eps_mean = exact_year.pivot(index='trade_date', columns='ts_code', values='eps_roll_mean') \
                    .reindex(index=dates, columns=universe)
    delta_eps = sum([eps_mean.pct_change(periods=63).fillna(0).shift(i*63).div(i+1) for i in range(4)]).loc[start_date:end_date]

    roll_cnt = exact_year.pivot(index='trade_date', columns='ts_code', values='roll_cnt') \
                    .reindex(index=dates, columns=universe)
    delta_cnt = sum([roll_cnt.pct_change(periods=21).fillna(0).shift(i*21).div(i+1) for i in range(4)]).loc[start_date:end_date]
    
    # ================================================================
    # 组装输出
    # ================================================================
    print('组装结果...')

    factor_dfs = [daily_std, cum_range, st_rev, season, indmom_mat, beta, hist_sigma, alpha, rs, 
                  long_term_rs, long_term_alpha, lncap, midcap, stom, stoq, stoa, atvr, mlev, blev, dtoa, 
                  btp, etp, cftp, ebit_ev, dp, var_sales, var_earnings, var_cashflows, accr_bs, accr_cf, 
                  ato, gp, gpm, roa, ta_growth, issuance_growth, capex_growth, eps_growth, sps_growth, 
                  sdafep, apebs, pred_growth3, adtp, delta_apebs, delta_eps, delta_cnt]
    factor_names = ['daily_std','cum_range', 'st_rev', 'season', 'indmom_mat', 'beta', 'hist_sigma', 'alpha', 
                    'rs', 'long_term_rs', 'long_term_alpha', 'lncap','midcap', 'stom', 'stoq', 'stoa', 'atvr', 
                    'mlev', 'blev', 'dtoa', 'btp', 'etp','cftp','ebit_ev', 'dp', 'var_sales','var_earnings',
                    'var_cashflows','accr_bs','accr_cf','ato','gp','gpm','roa', 'ta_growth','issuance_growth',
                    'capex_growth','eps_growth','sps_growth','sdafep','apebs','pred_growth3',
                    'adtp','delta_apebs','delta_eps','delta_cnt']
    
    result = pd.concat(factor_dfs, axis=1, keys=factor_names).stack(level=1).reset_index(
        names=["trade_date", "ts_code"])
    print('done')
    return result


# ── 使用示例 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    allstocks = pd.read_csv('data/allstock.csv')
    allstocks = allstocks[~allstocks['ts_code'].str.contains('BJ')].ts_code.tolist()

    df = calc_cne6_factors(start_date='2024-01-01', end_date='2025-12-31', allstocks=allstocks)
    
    
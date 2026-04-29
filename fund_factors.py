"""
CNE-6 基本面因子计算
换手率、市值、杠杆、估值、分红、财务质量、投资、增长、分析师预测
"""

import numpy as np
import pandas as pd
from numba import njit
from datatools import get_basic, get_finance, get_finance_ttm, get_report_roll


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def make_weights(window, half_life):
    tau = window / half_life * np.log(2)
    w = np.exp(-tau * (1 - np.arange(window) / window))
    return w / w.sum()


@njit(cache=True)
def _ewrs_core(arr_clean, nan_f, w0, q, qk, k, T, N):
    V = np.empty((T, N))
    V[0] = arr_clean[0]
    C = np.empty((T, N))
    C[0] = nan_f[0]
    for t in range(1, T):
        v = arr_clean[t] + q * V[t - 1]
        c = nan_f[t] + C[t - 1]
        if t >= k:
            v -= qk * arr_clean[t - k]
            c -= nan_f[t - k]
        V[t] = v
        C[t] = c
    return V, C


def exp_wt_rolling_sum(arr, w):
    """滚动加权求和（numba IIR 递推版，O(T×N)）"""
    T, N = arr.shape
    k = len(w)
    w_rev = w[::-1]
    q = w_rev[1] / w_rev[0]
    qk = q ** k
    nan_mask = np.isnan(arr)
    arr_clean = np.where(nan_mask, 0.0, arr).astype(np.float64)
    nan_f = nan_mask.astype(np.float64)
    V, C = _ewrs_core(arr_clean, nan_f, w_rev[0], q, qk, k, T, N)
    out = w_rev[0] * V
    out[C > 0] = np.nan
    out[:k - 1] = np.nan
    return out


@njit(cache=True)
def _slope_5y_core(mat, target_years, annual_years, yc5, ss5, T, N):
    """numba 核心：5年回归斜率/均值"""
    result = np.full((T, N), np.nan)
    n_annual = len(annual_years)
    for i in range(T):
        # 找到最后5个 <= target_years[i] 的年度索引
        count = 0
        last5 = np.empty(5, dtype=np.int64)
        for j in range(n_annual):
            if annual_years[j] <= target_years[i]:
                last5[count % 5] = j
                count += 1
        if count >= 5:
            # 取最后5个
            start = count - 5
            idx = np.empty(5, dtype=np.int64)
            for s in range(5):
                idx[s] = last5[(start + s) % 5]
            for n in range(N):
                slope = 0.0
                mean = 0.0
                valid = True
                for s in range(5):
                    val = mat[idx[s], n]
                    if np.isnan(val):
                        valid = False
                        break
                    slope += yc5[s] * val
                    mean += val
                if valid:
                    slope /= ss5
                    mean /= 5.0
                    if np.abs(mean) > 1e-8:
                        result[i, n] = slope / mean
    return result


def slope_5y(mat, annual_dates, target_dates):
    """
    过去5个财年对时间回归的斜率 / 均值（numba 加速）
    mat: (n_annual, N) 年度数据
    annual_dates: DatetimeIndex 年度日期
    target_dates: 目标日期数组
    返回 (T, N)
    """
    yc5 = np.arange(5, dtype=np.float64) - 2.0
    ss5 = (yc5 ** 2).sum()
    annual_years = pd.DatetimeIndex(annual_dates).year.to_numpy(dtype=np.float64)
    target_years = pd.DatetimeIndex(target_dates).year.to_numpy(dtype=np.float64)
    return _slope_5y_core(np.ascontiguousarray(mat, dtype=np.float64),
                          target_years, annual_years, yc5, ss5,
                          len(target_dates), mat.shape[1])


@njit(cache=True)
def _variation_5y_core(arr, annual_years, target_years, T, N):
    """numba 核心：5年波动率"""
    result = np.full((T, N), np.nan)
    n_annual = len(annual_years)
    for i in range(T):
        count = 0
        last5 = np.empty(5, dtype=np.int64)
        for j in range(n_annual):
            if annual_years[j] <= target_years[i]:
                last5[count % 5] = j
                count += 1
        if count >= 5:
            start = count - 5
            idx = np.empty(5, dtype=np.int64)
            for s in range(5):
                idx[s] = last5[(start + s) % 5]
            for n in range(N):
                mean = 0.0
                valid_count = 0
                for s in range(5):
                    val = arr[idx[s], n]
                    if not np.isnan(val):
                        mean += val
                        valid_count += 1
                if valid_count >= 5:
                    mean /= 5.0
                    if np.abs(mean) > 1e-6:
                        var = 0.0
                        for s in range(5):
                            diff = arr[idx[s], n] - mean
                            var += diff * diff
                        var /= 4.0  # ddof=1
                        result[i, n] = np.sqrt(var) / np.abs(mean)
    return result


def variation_5y(annual_df, dates, universe):
    """5年波动率（numba 加速）"""
    annual_years = pd.DatetimeIndex(annual_df.index).year.to_numpy(dtype=np.float64)
    target_years = pd.DatetimeIndex(dates).year.to_numpy(dtype=np.float64)
    arr = np.ascontiguousarray(annual_df.reindex(columns=universe).values, dtype=np.float64)
    return _variation_5y_core(arr, annual_years, target_years, len(dates), arr.shape[1])


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
    basic = get_basic(start_date=load_start, end_date=end_date,
                      fields=['circ_mv', 'pb', 'pe_ttm', 'dv_ttm', 'turnover_rate'])
    basic = basic[basic['ts_code'].isin(allstocks)].copy()
    basic['trade_date'] = pd.to_datetime(basic['trade_date'])

    print('加载财务数据...')
    fin = get_finance(start_date=fin_start, end_date=end_date, fields=[
        'total_assets', 'total_liab', 'total_ncl', 'non_cur_liab_due_1y',
        'lt_borr', 'st_borr', 'total_share', 'basic_eps', 'total_revenue',
        'n_income', 'n_income_attr_p', 'n_incr_cash_cash_equ', 'total_cogs',
        'ebit', 'minority_int', 'total_hldr_eqy_exc_min_int',
        'oth_eqt_tools_p_shr', 'depr_fa_coga_dpba', 'amort_intang_assets',
        'lt_amort_deferred_exp', 'n_cashflow_act', 'n_cashflow_inv_act',
        'c_pay_acq_const_fiolta', 'c_cash_equ_end_period',
    ])
    fin = fin[fin['ts_code'].isin(allstocks)].copy()

    print('加载TTM数据...')
    fin_ttm = get_finance_ttm(start_date=ttm_start, end_date=end_date, fields=[
        'revenue_ttm', 'total_revenue_ttm', 'n_income_ttm', 'n_income_attr_p_ttm',
        'n_incr_cash_cash_equ_ttm', 'total_cogs_ttm', 'ebit_ttm',
        'depr_fa_coga_dpba_ttm', 'amort_intang_assets_ttm', 'lt_amort_deferred_exp_ttm',
        'n_cashflow_act_ttm', 'n_cashflow_inv_act_ttm',
        'c_pay_acq_const_fiolta_ttm', 'c_cash_equ_end_period_ttm', 'basic_eps_ttm'])
    fin_ttm = fin_ttm[fin_ttm['ts_code'].isin(allstocks)].copy()
    fin_ttm = fin_ttm.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

    print('加载分析师预测数据...')
    predict = get_report_roll(year=int(start_date[:4]),
                              start_date=load_start, end_date=end_date)

    # ── 构建基础矩阵 ─────────────────────────────────────────────────────────
    dates = np.sort(basic['trade_date'].unique())
    universe = np.sort(basic['ts_code'].unique())
    n_dates, n_stocks = len(dates), len(universe)
    print(f'矩阵: {n_dates} 交易日 × {n_stocks} 只股票')

    # basic 数据 pivot
    circ_mv = basic.pivot(index='trade_date', columns='ts_code', values='circ_mv') \
                    .reindex(index=dates, columns=universe).values  # 单位：万元
    pb = basic.pivot(index='trade_date', columns='ts_code', values='pb') \
              .reindex(index=dates, columns=universe).values
    pe_ttm = basic.pivot(index='trade_date', columns='ts_code', values='pe_ttm') \
                   .reindex(index=dates, columns=universe).values
    dv_ttm = basic.pivot(index='trade_date', columns='ts_code', values='dv_ttm') \
                   .reindex(index=dates, columns=universe).values
    turnover = basic.pivot(index='trade_date', columns='ts_code', values='turnover_rate') \
                     .reindex(index=dates, columns=universe).values

    # 索引映射（用于快速 pivot）
    date_idx = pd.Series(np.arange(len(dates)), index=dates)
    stock_idx = pd.Series(np.arange(len(universe)), index=universe)

    # TTM 数据 pivot — numpy 索引
    ttm_fields = ['revenue_ttm', 'total_revenue_ttm', 'n_income_attr_p_ttm',
                  'n_cashflow_act_ttm', 'total_cogs_ttm', 'ebit_ttm',
                  'depr_fa_coga_dpba_ttm', 'amort_intang_assets_ttm', 'lt_amort_deferred_exp_ttm',
                  'c_pay_acq_const_fiolta_ttm', 'n_cashflow_inv_act_ttm']
    ttm_di = fin_ttm['trade_date'].map(date_idx).values
    ttm_si = fin_ttm['ts_code'].map(stock_idx).values
    ttm_valid = ~(np.isnan(ttm_di) | np.isnan(ttm_si))
    ttm_di = ttm_di[ttm_valid].astype(int)
    ttm_si = ttm_si[ttm_valid].astype(int)

    ttm_arrays = {}
    for field in ttm_fields:
        arr = np.full((n_dates, n_stocks), np.nan)
        vals = fin_ttm[field].values[ttm_valid]
        arr[ttm_di, ttm_si] = vals
        ttm_arrays[field] = arr

    rev_ttm = ttm_arrays['revenue_ttm']
    ni_ttm = ttm_arrays['n_income_attr_p_ttm']
    cfo_ttm = ttm_arrays['n_cashflow_act_ttm']
    cogs_ttm = ttm_arrays['total_cogs_ttm']
    ebit_ttm = ttm_arrays['ebit_ttm']
    da_ttm = ttm_arrays['depr_fa_coga_dpba_ttm'] + ttm_arrays['amort_intang_assets_ttm'] + ttm_arrays['lt_amort_deferred_exp_ttm']
    capex_ttm = ttm_arrays['c_pay_acq_const_fiolta_ttm']
    cfi_ttm = ttm_arrays['n_cashflow_inv_act_ttm']

    # 资产负债表（最新值）— 用 numpy 索引替代逐字段 pivot
    fin_fields = ['total_assets', 'total_liab', 'total_ncl', 'non_cur_liab_due_1y',
                  'lt_borr', 'total_share', 'minority_int', 'total_hldr_eqy_exc_min_int',
                  'oth_eqt_tools_p_shr', 'basic_eps', 'n_income_attr_p',
                  'c_cash_equ_end_period', 'total_revenue', 'n_income', 'n_incr_cash_cash_equ']
    fin_di = fin['trade_date'].map(date_idx).values
    fin_si = fin['ts_code'].map(stock_idx).values
    valid_mask = ~(np.isnan(fin_di) | np.isnan(fin_si))
    fin_di = fin_di[valid_mask].astype(int)
    fin_si = fin_si[valid_mask].astype(int)

    fin_arrays = {}
    for field in fin_fields:
        arr = np.full((n_dates, n_stocks), np.nan)
        vals = fin[field].values[valid_mask]
        arr[fin_di, fin_si] = vals
        fin_arrays[field] = arr

    total_assets = fin_arrays['total_assets']
    total_liab = fin_arrays['total_liab']
    total_ncl = fin_arrays['total_ncl']
    ncl_due_1y = fin_arrays['non_cur_liab_due_1y']
    lt_borr = fin_arrays['lt_borr']
    total_share = fin_arrays['total_share']
    minority_int = fin_arrays['minority_int']
    eqt_exc_min = fin_arrays['total_hldr_eqy_exc_min_int']
    oth_eqt = fin_arrays['oth_eqt_tools_p_shr']
    basic_eps = fin_arrays['basic_eps']
    ni_attr_p = fin_arrays['n_income_attr_p']
    c_cash_end = fin_arrays['c_cash_equ_end_period']
    revenue = fin_arrays['total_revenue']
    n_income = fin_arrays['n_income']
    n_incr_cash = fin_arrays['n_incr_cash_cash_equ']

    # ================================================================
    # 换手率因子
    # ================================================================
    print('计算换手率因子...')
    stom = np.log(pd.DataFrame(turnover).rolling(21, min_periods=21).sum().values + 1)
    stoq_data = np.log(turnover + 1)
    stoq = np.log(pd.DataFrame(stoq_data).rolling(63, min_periods=63).mean().values * 63 + 1)
    stoa_data = np.log(turnover + 1)
    stoa = np.log(pd.DataFrame(stoa_data).rolling(252, min_periods=252).mean().values * 252 + 1)

    w_atvr = make_weights(252, 63)
    atvr = exp_wt_rolling_sum(turnover, w_atvr)

    # ================================================================
    # 市值因子
    # ================================================================
    print('计算市值因子...')
    lncap = np.log(circ_mv)
    midcap = lncap ** 3

    # ================================================================
    # 估值/杠杆因子
    # ================================================================
    print('计算估值/杠杆因子...')
    # 财务数据单位：元；circ_mv 单位：万元
    # 统一到万元
    total_assets_w = total_assets / 10000
    total_liab_w = total_liab / 10000
    total_ncl_w = total_ncl / 10000
    equity_w = total_assets_w - total_liab_w
    pe_bv_w = oth_eqt * total_share / 10000  # oth_eqt(元/股) * total_share(万股) / 10000
    minority_int_w = minority_int / 10000
    c_cash_end_w = c_cash_end / 10000
    # LD = total_ncl(非流动负债) 替代 lt_borr(lt_borr NaN 太多)
    ld_w = total_ncl_w

    me = circ_mv  # 万元

    mlev = np.clip((me + pe_bv_w + ld_w) / me, 0, 10)
    blev = np.clip((equity_w + pe_bv_w + ld_w) / me, 0, 10)
    dtoa = total_liab / total_assets

    btp = 1.0 / np.where(np.abs(pb) > 1e-6, pb, np.nan)
    etp = 1.0 / np.where(np.abs(pe_ttm) > 1e-6, pe_ttm, np.nan)
    cftp = ni_ttm / 10000 / me  # ni_ttm(元) / 10000 / me(万元)

    ev = me + total_ncl_w + pe_bv_w + minority_int_w - c_cash_end_w
    ebit_ev = ebit_ttm / 10000 / ev  # ebit_ttm(元) / 10000 / ev(万元)

    dp = dv_ttm / 100.0  # 已是百分比

    # ================================================================
    # 财务质量因子（5年波动率）
    # ================================================================
    print('计算财务质量因子...')
    # 提取年度数据（Q4 报告）
    fin['stat_date'] = pd.to_datetime(fin['stat_date'])
    # 每个 ts_code+trade_date 只保留最新报告（按 stat_date 降序取第一条）
    fin = fin.sort_values(['ts_code', 'trade_date', 'stat_date'], ascending=[True, True, False])
    fin = fin.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
    q4_mask = fin['stat_date'].dt.month == 12
    fin_annual = fin[q4_mask].copy()

    # 年度 pivot
    fin_annual = fin_annual.drop_duplicates(subset=['ts_code', 'stat_date'], keep='last')

    def annual_pivot(field):
        df = fin_annual.pivot(index='stat_date', columns='ts_code', values=field)
        return df

    rev_annual = annual_pivot('total_revenue')
    ni_annual = annual_pivot('n_income')
    cash_annual = annual_pivot('n_incr_cash_cash_equ')

    # ACCR_BS 用的年度数组
    ta_annual_df = annual_pivot('total_assets').reindex(columns=universe)
    tl_annual_df = annual_pivot('total_liab').reindex(columns=universe)
    ncl_annual_df = annual_pivot('total_ncl').reindex(columns=universe)
    cash_end_annual_df = annual_pivot('c_cash_equ_end_period').reindex(columns=universe)
    ta_annual_arr = ta_annual_df.values
    tl_annual_arr = tl_annual_df.values
    ncl_annual_arr = ncl_annual_df.values
    cash_annual_arr = cash_end_annual_df.values
    annual_dates = ta_annual_df.index

    # 5年波动率（numba 加速）
    var_sales = variation_5y(rev_annual, dates, universe)
    var_earnings = variation_5y(ni_annual, dates, universe)
    var_cashflows = variation_5y(cash_annual, dates, universe)

    # ================================================================
    # 应计项目
    # ================================================================
    print('计算应计项目...')
    # NOA = TA - Cash - (TL - TD)
    # TD(带息债务) ≈ total_ncl（近似：非流动负债大部分为带息）
    noa_annual = ta_annual_arr - cash_annual_arr - tl_annual_arr + ncl_annual_arr

    # ACCR_BS = -(NOA_t - NOA_{t-1}) / TA_t，使用年度Q4报告间的差分
    accr_bs = np.full((n_dates, n_stocks), np.nan)
    for i, d in enumerate(dates):
        valid = [j for j, yd in enumerate(annual_dates) if yd <= d]
        if len(valid) >= 2:
            curr, prev = valid[-1], valid[-2]
            d_noa = noa_annual[curr] - noa_annual[prev]
            accr_bs[i] = -d_noa / np.where(np.abs(ta_annual_arr[curr]) > 1e-6, ta_annual_arr[curr], np.nan)

    accr_cf = -(ni_ttm - cfo_ttm + da_ttm) / np.where(np.abs(total_assets) > 1e-6, total_assets, np.nan)

    # ================================================================
    # 盈利能力
    # ================================================================
    print('计算盈利能力...')
    # total_revenue_ttm 比 revenue_ttm 更完整
    rev_ttm_full = ttm_arrays['total_revenue_ttm']
    ato = rev_ttm_full / total_assets
    gp = (rev_ttm_full - cogs_ttm) / total_assets
    gpm = (rev_ttm_full - cogs_ttm) / np.where(np.abs(rev_ttm_full) > 1e-6, rev_ttm_full, np.nan)
    roa = ni_ttm / total_assets

    # ================================================================
    # 投资质量（5年增长率）
    # ================================================================
    print('计算投资质量...')
    ta_annual = annual_pivot('total_assets')
    ts_annual = annual_pivot('total_share')
    capex_annual = annual_pivot('c_pay_acq_const_fiolta')
    eps_annual = annual_pivot('basic_eps')
    rev_per_share_annual = annual_pivot('total_revenue')  # 需要除以 total_share

    ta_growth = -slope_5y(ta_annual.reindex(columns=universe).values,
                          ta_annual.index, dates)
    issuance_growth = -slope_5y(ts_annual.reindex(columns=universe).values,
                                ts_annual.index, dates)
    capex_growth = -slope_5y(capex_annual.reindex(columns=universe).values,
                             capex_annual.index, dates)

    eps_growth = slope_5y(eps_annual.reindex(columns=universe).values,
                          eps_annual.index, dates)
    # 每股营收 = 总营收 / 总股本
    rps_annual_arr = rev_annual.reindex(columns=universe).values / \
                     np.where(ts_annual.reindex(columns=universe).values > 0,
                              ts_annual.reindex(columns=universe).values, np.nan)
    sps_growth = slope_5y(rps_annual_arr, rev_annual.index, dates)

    # ================================================================
    # 分析师预测因子
    # ================================================================
    print('计算分析师预测因子...')
    pred = predict.copy()
    pred['trade_date'] = pd.to_datetime(pred['trade_date'])
    pred = pred.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

    # eps_std / close → sdafep
    eps_std = pred.pivot(index='trade_date', columns='ts_code', values='eps_roll_std') \
                   .reindex(index=dates, columns=universe).values
    # close = circ_mv(万元) * 10000 / total_share(股) = 元/股
    close_proxy = circ_mv * 10000 / np.where(total_share > 0, total_share, np.nan)
    sdafep = eps_std / np.where(np.abs(close_proxy) > 0.01, close_proxy, np.nan)

    eps_mean = pred.pivot(index='trade_date', columns='ts_code', values='eps_roll_mean') \
                    .reindex(index=dates, columns=universe).values
    np_mean = pred.pivot(index='trade_date', columns='ts_code', values='np_roll_mean') \
                   .reindex(index=dates, columns=universe).values
    # np_mean(万元) / me(万元) = 无量纲
    apebs = np_mean / me

    rd_mean = pred.pivot(index='trade_date', columns='ts_code', values='rd_roll_mean') \
                   .reindex(index=dates, columns=universe).values
    # rd_mean(元/股) / close(元/股)
    adtp = rd_mean / np.where(np.abs(close_proxy) > 0.01, close_proxy, np.nan)

    # 预测3年增长率
    np_std = pred.pivot(index='trade_date', columns='ts_code', values='np_roll_std') \
                  .reindex(index=dates, columns=universe).values
    pred_growth3 = np_mean / np.where(np.abs(np_std) > 1e-6, np_std, np.nan)  # 简化

    # Revision ratio: (up - down) / total — 需要原始报告数据，暂用 roll_cnt 近似
    roll_cnt = pred.pivot(index='trade_date', columns='ts_code', values='roll_cnt') \
                    .reindex(index=dates, columns=universe).values
    # 暂不实现 revision_ratio（需要原始 up/down 数据）

    # Change in APEBS: 63日变化
    apebs_df = pd.DataFrame(apebs, index=dates, columns=universe)
    delta_apebs = apebs_df.diff(63).values / apebs_df.shift(63).values

    # Change in predicted EPS: 63日变化
    eps_mean_df = pd.DataFrame(eps_mean, index=dates, columns=universe)
    delta_eps = eps_mean_df.diff(63).values / eps_mean_df.shift(63).values

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
        'DELTA_EPS': delta_eps.ravel(),
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
